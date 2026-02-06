
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { GeminiModel, FigureExtractionResult, GridConfig, BoundingBox } from "../types";

export const extractDataFromFigure = async (
  apiKey: string,
  model: GeminiModel,
  prompt: string,
  imageParts: { data: string; mimeType: string; label?: string }[],
  gridConfig: GridConfig
): Promise<FigureExtractionResult> => {
  const ai = new GoogleGenAI({ apiKey });
  
  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      title: { type: Type.STRING, description: "Title of the scientific figure or chart." },
      xAxis: {
        type: Type.OBJECT,
        properties: {
          label: { type: Type.STRING, description: "Label of the X axis." },
          unit: { type: Type.STRING, description: "Unit of measurement for X axis if available." }
        },
        required: ["label"]
      },
      yAxis: {
        type: Type.OBJECT,
        properties: {
          label: { type: Type.STRING, description: "Label of the Y axis." },
          unit: { type: Type.STRING, description: "Unit of measurement for Y axis if available." }
        },
        required: ["label"]
      },
      dataPoints: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            label: { type: Type.STRING, description: "Legend label or category for this point." },
            xValue: { type: Type.NUMBER, description: "Numerical value on X axis." },
            yValue: { type: Type.NUMBER, description: "Numerical value on Y axis." }
          },
          required: ["label", "xValue", "yValue"]
        }
      },
      summary: { type: Type.STRING, description: "Brief description of the findings in the figure." }
    },
    required: ["title", "xAxis", "yAxis", "dataPoints", "summary"]
  };

  const contentsParts: any[] = [{ text: prompt }];

  imageParts.forEach((part) => {
    if (part.label) {
      contentsParts.push({ text: `Sub-image section: ${part.label}` });
    }
    contentsParts.push({
      inlineData: {
        data: part.data,
        mimeType: part.mimeType
      }
    });
  });

  const isGemini3 = model.startsWith('gemini-3');
  const config: any = {
    responseMimeType: "application/json",
    responseSchema: responseSchema,
  };

  if (isGemini3) {
    config.thinkingConfig = { thinkingBudget: 4000 };
  }

  const response: GenerateContentResponse = await ai.models.generateContent({
    model: model,
    contents: { parts: contentsParts },
    config: config
  });

  if (!response.text) {
    throw new Error("No response from model");
  }

  return JSON.parse(response.text) as FigureExtractionResult;
};

export const detectSubplot = async (
  apiKey: string,
  model: GeminiModel,
  image: string,
  query: string
): Promise<BoundingBox> => {
  const ai = new GoogleGenAI({ apiKey });
  
  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      ymin: { type: Type.NUMBER, description: "Normalized coordinate (0-1000)" },
      xmin: { type: Type.NUMBER, description: "Normalized coordinate (0-1000)" },
      ymax: { type: Type.NUMBER, description: "Normalized coordinate (0-1000)" },
      xmax: { type: Type.NUMBER, description: "Normalized coordinate (0-1000)" },
    },
    required: ["ymin", "xmin", "ymax", "xmax"]
  };

  const response = await ai.models.generateContent({
    model: model,
    contents: {
      parts: [
        { text: `Locate the subplot related to "${query}" in this multi-panel figure. 
        INSTRUCTIONS: 
        1. Identify the specific panel/subplot containing the requested information.
        2. Expand the bounding box to capture all contextual elements: axis labels (X and Y), numerical ticks, units (e.g., cm, K, s), legends, and any titles.
        3. Do not cut off the bottom edge where X-axis units are usually placed.
        Return the bounding box as normalized coordinates [0-1000] for ymin, xmin, ymax, xmax.` },
        { inlineData: { data: image.split(',')[1], mimeType: 'image/jpeg' } }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: responseSchema
    }
  });

  return JSON.parse(response.text!) as BoundingBox;
};
