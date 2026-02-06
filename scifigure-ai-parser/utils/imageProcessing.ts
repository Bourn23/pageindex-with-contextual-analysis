
import { GridConfig, BoundingBox } from "../types";

export const sliceImageIntoGrid = async (
  imageSrc: string,
  config: GridConfig
): Promise<{ data: string; mimeType: string; label: string }[]> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const patches: { data: string; mimeType: string; label: string }[] = [];
      const { rows, cols } = config;
      const patchWidth = img.width / cols;
      const patchHeight = img.height / rows;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const canvas = document.createElement('canvas');
          canvas.width = patchWidth;
          canvas.height = patchHeight;
          const ctx = canvas.getContext('2d');
          if (!ctx) continue;

          ctx.drawImage(
            img,
            c * patchWidth, r * patchHeight, patchWidth, patchHeight,
            0, 0, patchWidth, patchHeight
          );

          const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
          patches.push({
            data: base64,
            mimeType: 'image/jpeg',
            label: `Grid [Row ${r}, Col ${c}]`
          });
        }
      }
      resolve(patches);
    };
    img.onerror = reject;
    img.src = imageSrc;
  });
};

export const cropImage = async (
  imageSrc: string,
  box: BoundingBox
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const x = (box.xmin / 1000) * img.width;
      const y = (box.ymin / 1000) * img.height;
      const w = ((box.xmax - box.xmin) / 1000) * img.width;
      const h = ((box.ymax - box.ymin) / 1000) * img.height;
      
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return reject('No context');

      ctx.drawImage(img, x, y, w, h, 0, 0, w, h);
      resolve(canvas.toDataURL('image/jpeg', 0.9));
    };
    img.onerror = reject;
    img.src = imageSrc;
  });
};

export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = error => reject(error);
  });
};

export const parseCSV = (text: string): Record<string, any>[] => {
  const lines = text.split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).filter(line => line.trim()).map(line => {
    const values = line.split(',');
    return headers.reduce((obj, header, i) => {
      obj[header] = values[i]?.trim();
      return obj;
    }, {} as any);
  });
};
