import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

class SciFigureParser:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", debug: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.debug = debug
        self.client_v1 = genai.Client(api_key=api_key)
        self.client_alpha = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    def _is_gemini_3(self) -> bool:
        return "gemini-3" in self.model_name

    def _get_client(self):
        return self.client_alpha if self._is_gemini_3() else self.client_v1

    def _get_debug_path(self, image_path: str, suffix: str, ext: Optional[str] = None) -> str:
        """
        Creates a 'scifig_debug' folder in the image_path directory and returns the debug path.
        Avoids nested 'scifig_debug' folders.
        """
        dir_name = os.path.abspath(os.path.dirname(image_path))
        base_name = os.path.basename(image_path)
        file_name, original_ext = os.path.splitext(base_name)
        
        if os.path.basename(dir_name) == "scifig_debug":
            debug_dir = dir_name
        else:
            debug_dir = os.path.join(dir_name, "scifig_debug")
            
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
            
        extension = ext if ext else original_ext
        if not extension.startswith('.'):
            extension = f".{extension}"
            
        return os.path.join(debug_dir, f"{file_name}{suffix}{extension}")

    def _get_image_part(self, data: bytes, mime_type: str = "image/jpeg") -> types.Part:
        if self._is_gemini_3():
            return types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=data,
                ),
                media_resolution={"level": "media_resolution_high"}
            )
        else:
            return types.Part.from_bytes(data=data, mime_type=mime_type)

    def detect_subplot(self, image_path: str, query: str) -> Dict[str, float]:
        """
        Locates a subplot in a multi-panel figure based on a query.
        Returns normalized coordinates: ymin, xmin, ymax, xmax (0-1000).
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = f"""Identify and locate all subplots/panels related to "{query}" in this figure. Avoid schematics, graphics, and other non-numeric data.
        INSTRUCTIONS: 
        1. Determine if this is a multi-panel figure (e.g., contains 'Fig 1a', 'Fig 1b' OR separate sub-charts).
        2. For EVERY relevant subplot (even if there are multiple of the same type like two ionic conductivity plots):
           - Provide a bounding box that captures axis labels, ticks, units, and legends.
           - Assign a short label (e.g., 'A', 'B', 'Plot 1').
        3. If there is only ONE plot total in the entire image, set "is_multi_plot" to false.
        4. Return the result in a structured JSON format."""

        response = self._get_client().models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        self._get_image_part(image_data)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_multi_plot": {"type": "BOOLEAN", "description": "True if the figure contains multiple distinct subplots or panels."},
                        "detections": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "ymin": {"type": "NUMBER"},
                                    "xmin": {"type": "NUMBER"},
                                    "ymax": {"type": "NUMBER"},
                                    "xmax": {"type": "NUMBER"},
                                    "label": {"type": "STRING", "description": "Label for this specific subplot."}
                                },
                                "required": ["ymin", "xmin", "ymax", "xmax"]
                            }
                        },
                        "isIonicConductivity": {"type": "BOOLEAN", "description": "True if the figure contains ionic conductivity measurements."}
                    },
                    "required": ["is_multi_plot", "detections", "isIonicConductivity"]
                }
            )
        )
        
        result = json.loads(response.text)

        if self.debug:
            self._visualize_detection(image_path, result.get('detections', []), query)

        return result

    def _visualize_detection(self, image_path: str, detections: List[Dict[str, Any]], query: str):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        for i, box in enumerate(detections):
            left = (box['xmin'] / 1000) * w
            top = (box['ymin'] / 1000) * h
            right = (box['xmax'] / 1000) * w
            bottom = (box['ymax'] / 1000) * h
            
            color = "red" if i == 0 else "blue" # Alternate colors if helpful
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            label = box.get('label', f'Plot {i+1}')
            draw.text((left + 10, top + 10), label, fill=color)
        
        debug_path = self._get_debug_path(image_path, "_debug_detection", ext="png")
        img.save(debug_path)
        print(f"[DEBUG] Detection visualization saved to {debug_path}")

    def crop_image(self, image_path: str, box: Dict[str, float], output_path: str = None, padding: int = 0) -> str:
        """
        Crops an image based on normalized coordinates (0-1000).
        Automatically adds safety padding if specified.
        """
        img = Image.open(image_path)
        width, height = img.size

        # Apply padding to normalized coordinates
        ymin = max(0, box['ymin'] - padding / 2)
        xmin = max(0, box['xmin'] - padding / 2)
        ymax = min(1000, box['ymax'] + padding)
        xmax = min(1000, box['xmax'] + padding)

        left = (xmin / 1000) * width
        top = (ymin / 1000) * height
        right = (xmax / 1000) * width
        bottom = (ymax / 1000) * height

        cropped_img = img.crop((left, top, right, bottom))
        
        if output_path is None:
            output_path = self._get_debug_path(image_path, "_cropped")
            
        cropped_img.save(output_path)
        
        if self.debug:
            print(f"[DEBUG] Cropped image saved to {output_path}")
            
        return output_path

    def _slice_image(self, image_path: str, rows: int, cols: int) -> List[Dict[str, Any]]:
        """
        Slices an image into a grid of patches for grounding.
        """
        img = Image.open(image_path)
        width, height = img.size
        patch_width = width // cols
        patch_height = height // rows
        
        patches = []
        for r in range(rows):
            for c in range(cols):
                left = c * patch_width
                top = r * patch_height
                right = (c + 1) * patch_width
                bottom = (r + 1) * patch_height
                
                patch = img.crop((left, top, right, bottom))
                buffer = BytesIO()
                patch.save(buffer, format="JPEG")
                patch_data = buffer.getvalue()
                
                patches.append({
                    "data": patch_data,
                    "label": f"Grid [Row {r}, Col {c}]"
                })
        return patches

    def extract_data(self, image_path: str, grid_config: Optional[Dict[str, Any]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts structured data from a scientific figure.
        Supports grid grounding by slicing the image into patches.
        """
        if prompt is None:
            prompt = """Analyze this scientific figure in high detail. 
            1. Identify the axes, their units, and the scale type (linear or log).
            2. Extract numerical data points from the chart/plot.
            3. Verify if this plot represents ionic conductivity measurements. 
               - Especially check if the X-axis is temperature or inverse temperature (e.g., 1000/T).
               - If temperature is not explicitly specified, assume the measurement was taken at room temperature (25 degrees Celsius) if it's a conductivity measurement.
            4. If multiple series exist, distinguish them using labels.
            5. If grid patches are provided, use them to verify tick marks and small text details.
            Return the result in structured JSON format."""

        with open(image_path, "rb") as f:
            original_image_data = f.read()

        parts = [types.Part(text=prompt), self._get_image_part(original_image_data)]

        patches = []
        if grid_config and grid_config.get("enabled"):
            rows = grid_config.get("rows", 2)
            cols = grid_config.get("cols", 2)
            patches = self._slice_image(image_path, rows, cols)
            for patch in patches:
                parts.append(types.Part(text=f"Sub-image section: {patch['label']}"))
                parts.append(self._get_image_part(patch['data']))
        
        if self.debug:
            # save the image
            img_type = image_path.split(".")[-1]
            if grid_config and grid_config.get("enabled"):
                for patch in patches:
                    patch_name = patch['label'].replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
                    save_path = self._get_debug_path(image_path, f"_debug_extraction_input_{patch_name}")
                    with open(save_path, "wb") as f:
                        f.write(patch['data'])
            else:
                save_path = self._get_debug_path(image_path, "_debug_extraction_input")
                with open(save_path, "wb") as f:
                    f.write(original_image_data)

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "Title of the scientific figure or chart."},
                "xAxis": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "description": "Label of the X axis."},
                        "unit": {"type": "STRING", "description": "Unit of measurement for X axis if available."},
                        "scale": {"type": "STRING", "enum": ["linear", "log"], "description": "Scale type of the X axis."}
                    },
                    "required": ["label", "scale"]
                },
                "yAxis": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "description": "Label of the Y axis."},
                        "unit": {"type": "STRING", "description": "Unit of measurement for Y axis if available."},
                        "scale": {"type": "STRING", "enum": ["linear", "log"], "description": "Scale type of the Y axis."}
                    },
                    "required": ["label", "scale"]
                },
                "dataPoints": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "label": {"type": "STRING", "description": "Legend label or category for this point."},
                            "xValue": {"type": "NUMBER", "description": "Numerical value on X axis."},
                            "yValue": {"type": "NUMBER", "description": "Numerical value on Y axis."}
                        },
                        "required": ["label", "xValue", "yValue"]
                    }
                },
                "summary": {"type": "STRING", "description": "Brief description of the findings in the figure."}
            },
            "required": ["title", "xAxis", "yAxis", "dataPoints", "summary"]
        }

        response = self._get_client().models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )

        result = json.loads(response.text)
        
        if self.debug:
            self._visualize_extraction(image_path, result)
            
        return result

    def _visualize_extraction(self, image_path: str, result: Dict[str, Any]):
        """
        Plots the extracted data side-by-side with the original input figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot original image
        img = Image.open(image_path)
        ax1.imshow(img)
        ax1.set_title("Input Figure to Extraction")
        ax1.axis('off')
        
        # Plot extracted data
        data_points = result.get('dataPoints', [])
        labels = list(set(dp['label'] for dp in data_points))
        
        for label in labels:
            x = [dp['xValue'] for dp in data_points if dp['label'] == label]
            y = [dp['yValue'] for dp in data_points if dp['label'] == label]
            ax2.scatter(x, y, label=label)
            
        x_label = result.get('xAxis', {}).get('label', 'X')
        x_unit = result.get('xAxis', {}).get('unit', '')
        x_scale = result.get('xAxis', {}).get('scale', 'linear')
        
        y_label = result.get('yAxis', {}).get('label', 'Y')
        y_unit = result.get('yAxis', {}).get('unit', '')
        y_scale = result.get('yAxis', {}).get('scale', 'linear')
        
        ax2.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
        ax2.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
        ax2.set_title(result.get('title', 'Extracted Data'))
        
        if x_scale == 'log':
            ax2.set_xscale('log')
        if y_scale == 'log':
            ax2.set_yscale('log')
            
        if labels:
            ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7, which="both")
        
        debug_path = self._get_debug_path(image_path, "_debug_extraction", ext="png")
        plt.tight_layout()
        plt.savefig(debug_path)
        plt.close()
        print(f"[DEBUG] Extraction visualization saved to {debug_path}")

if __name__ == "__main__":
    # Example usage (requires API_KEY in env)
    import sys
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    parser = SciFigureParser(api_key=api_key, debug=True)
    # figure_path = "path/to/your/figure.jpg"
    # box = parser.detect_subplot(figure_path, "ionic conductivity")
    # cropped_path = parser.crop_image(figure_path, box, padding=80)
    # result = parser.extract_data(cropped_path, grid_config={"enabled": True, "rows": 2, "cols": 2})
    # print(json.dumps(result, indent=2))
