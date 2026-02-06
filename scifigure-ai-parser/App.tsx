
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { 
  FileSearch, 
  Settings, 
  BarChart3, 
  Upload, 
  Cpu, 
  LayoutGrid, 
  Loader2, 
  CheckCircle2, 
  AlertCircle,
  Table as TableIcon,
  RefreshCw,
  Crop as CropIcon,
  Search,
  Check,
  X,
  Maximize2,
  Minimize2,
  Move
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';
import { GeminiModel, FigureExtractionResult, GridConfig, BoundingBox, ActiveTab } from './types';
import { extractDataFromFigure, detectSubplot } from './services/geminiService';
import { fileToBase64, sliceImageIntoGrid, parseCSV, cropImage } from './utils/imageProcessing';

const DEFAULT_PROMPT = `Analyze this scientific figure in high detail. 
1. Identify the axes and their units.
2. Extract numerical data points from the chart/plot.
3. If multiple series exist, distinguish them using labels.
4. If a grid is provided, use the sub-images to verify tick marks and small text details.
Return the result in structured JSON format.`;

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('parser');
  const [model, setModel] = useState<GeminiModel>(GeminiModel.FLASH_3);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [image, setImage] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [csvData, setCsvData] = useState<any[] | null>(null);
  const [gridConfig, setGridConfig] = useState<GridConfig>({ enabled: true, rows: 2, cols: 2 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FigureExtractionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Cropper State
  const [cropQuery, setCropQuery] = useState('ionic conductivity');
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragType, setDragType] = useState<'move' | 'resize' | null>(null);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const base64 = await fileToBase64(file);
      setOriginalImage(base64);
      setImage(base64);
      setResult(null);
      setCurrentBox(null);
    }
  };

  const handleCsvUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const text = await file.text();
      const parsed = parseCSV(text);
      setCsvData(parsed);
    }
  };

  const handleAutoCrop = async () => {
    if (!originalImage) return;
    setLoading(true);
    setError(null);
    try {
      const apiKey = (process.env.API_KEY || '').trim();
      const box = await detectSubplot(apiKey, model, originalImage, cropQuery);
      setCurrentBox(box);
    } catch (err: any) {
      setError("AI failed to locate the subplot. Try a different query.");
    } finally {
      setLoading(false);
    }
  };

  const autoExpand = (padding: number = 80) => {
    if (!currentBox) return;
    // Normalized expansion to capture axis labels and ticks
    setCurrentBox({
      xmin: Math.max(0, currentBox.xmin - padding / 2),
      ymin: Math.max(0, currentBox.ymin - padding / 2),
      xmax: Math.min(1000, currentBox.xmax + padding),
      ymax: Math.min(1000, currentBox.ymax + padding), // Be more generous on bottom
    });
  };

  const applyCrop = async () => {
    if (!originalImage || !currentBox) return;
    try {
      const cropped = await cropImage(originalImage, currentBox);
      setImage(cropped);
      setActiveTab('parser');
      setResult(null);
    } catch (err) {
      setError("Failed to crop image.");
    }
  };

  const handleProcess = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    try {
      const originalBase64 = image.split(',')[1];
      let imageParts = [{ data: originalBase64, mimeType: 'image/jpeg' }];
      
      if (gridConfig.enabled) {
        const patches = await sliceImageIntoGrid(image, gridConfig);
        imageParts = [...imageParts, ...patches];
      }

      const apiKey = (process.env.API_KEY || '').trim();
      if (!apiKey) throw new Error("API Key not found in environment.");

      const extracted = await extractDataFromFigure(apiKey, model, prompt, imageParts, gridConfig);
      setResult(extracted);
    } catch (err: any) {
      setError(err.message || "An error occurred while processing.");
    } finally {
      setLoading(false);
    }
  };

  const handleMouseDown = (e: React.MouseEvent, type: 'move' | 'resize') => {
    if (!currentBox) return;
    e.stopPropagation();
    setIsDragging(true);
    setDragType(type);
    setStartPos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !currentBox || !imgRef.current) return;

    const dx = e.clientX - startPos.x;
    const dy = e.clientY - startPos.y;
    const { width, height } = imgRef.current.getBoundingClientRect();
    
    const ndx = (dx / width) * 1000;
    const ndy = (dy / height) * 1000;

    if (dragType === 'move') {
      setCurrentBox(prev => {
        if (!prev) return null;
        let nxMin = prev.xmin + ndx;
        let nyMin = prev.ymin + ndy;
        let nxMax = prev.xmax + ndx;
        let nyMax = prev.ymax + ndy;

        if (nxMin < 0) { nxMax -= nxMin; nxMin = 0; }
        if (nyMin < 0) { nyMax -= nyMin; nyMin = 0; }
        if (nxMax > 1000) { nxMin -= (nxMax - 1000); nxMax = 1000; }
        if (nyMax > 1000) { nyMin -= (nyMax - 1000); nyMax = 1000; }

        return { xmin: nxMin, ymin: nyMin, xmax: nxMax, ymax: nyMax };
      });
    } else if (dragType === 'resize') {
      setCurrentBox(prev => {
        if (!prev) return null;
        return {
          ...prev,
          xmax: Math.min(1000, Math.max(prev.xmin + 20, prev.xmax + ndx)),
          ymax: Math.min(1000, Math.max(prev.ymin + 20, prev.ymax + ndy))
        };
      });
    }

    setStartPos({ x: e.clientX, y: e.clientY });
  }, [isDragging, dragType, currentBox, startPos]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragType(null);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const comparisonData = useMemo(() => {
    if (!result || !csvData) return null;
    return result.dataPoints.map((dp, idx) => {
      const gt = csvData[idx];
      return {
        label: dp.label,
        extracted: typeof dp.yValue === 'number' ? dp.yValue : parseFloat(dp.yValue as string),
        groundTruth: gt ? parseFloat(gt.yValue || gt.value || gt[Object.keys(gt)[1]]) : 0
      };
    });
  }, [result, csvData]);

  return (
    <div className="flex flex-col h-screen text-slate-800 select-none overflow-hidden">
      {/* Header */}
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between shadow-sm sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <FileSearch className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">SciFigure AI Parser</h1>
            <p className="text-xs text-slate-500 font-medium">Extracting Knowledge from Visual Data</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center bg-slate-100 p-1 rounded-full px-3 gap-2 text-sm border">
            <Cpu className="w-4 h-4 text-blue-600" />
            <select 
              value={model} 
              onChange={(e) => setModel(e.target.value as GeminiModel)}
              className="bg-transparent font-bold text-blue-700 focus:outline-none cursor-pointer text-xs"
            >
              <option value={GeminiModel.FLASH_3}>Gemini 3 Flash</option>
              <option value={GeminiModel.PRO_3}>Gemini 3 Pro</option>
              <option value={GeminiModel.FLASH_2_5}>Gemini 2.5 Flash</option>
              <option value={GeminiModel.PRO_2_5}>Gemini 2.5 Pro</option>
            </select>
          </div>
          <div className="flex p-1 bg-slate-100 rounded-full border">
            <button 
              onClick={() => setActiveTab('parser')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${activeTab === 'parser' ? 'bg-white shadow text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Parser
            </button>
            <button 
              onClick={() => setActiveTab('cropper')}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all ${activeTab === 'cropper' ? 'bg-white shadow text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Cropper
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden flex bg-slate-50">
        <aside className="w-80 border-r bg-white overflow-y-auto p-6 space-y-8 shadow-inner flex-shrink-0">
          <section>
            <div className="flex items-center gap-2 mb-4 text-blue-600">
              <Upload className="w-5 h-5" />
              <h2 className="font-bold uppercase tracking-wider text-xs">Data Input</h2>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-slate-500 mb-1">IMAGE SOURCE</label>
                <div className="relative group">
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="absolute inset-0 opacity-0 w-full h-full cursor-pointer z-10" />
                  <div className="border-2 border-dashed border-slate-200 rounded-xl p-4 text-center group-hover:border-blue-300 transition-colors">
                    {image ? (
                      <div className="relative">
                        <img src={image} className="w-full h-32 object-contain rounded-md shadow-sm" alt="Thumbnail" />
                        <button onClick={() => {setImage(originalImage); setResult(null); setCurrentBox(null)}} className="absolute top-1 right-1 p-1 bg-white/80 rounded-full text-red-500 shadow hover:bg-white"><X className="w-3 h-3" /></button>
                      </div>
                    ) : (
                      <div className="py-4">
                        <Upload className="w-8 h-8 mx-auto text-slate-300 mb-2" />
                        <span className="text-xs text-slate-400 font-medium">Click to upload Figure</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-tighter">Ground Truth (CSV)</label>
                <input type="file" accept=".csv" onChange={handleCsvUpload} className="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition-all cursor-pointer" />
                {csvData && <p className="mt-2 text-[10px] text-green-600 font-bold flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Ground truth loaded</p>}
              </div>
            </div>
          </section>

          {activeTab === 'parser' ? (
            <section className="animate-in slide-in-from-left-4">
              <div className="flex items-center gap-2 mb-4 text-blue-600">
                <Settings className="w-5 h-5" />
                <h2 className="font-bold uppercase tracking-wider text-xs">Extraction Config</h2>
              </div>
              <div className="space-y-4">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">Grid Grounding</label>
                    <input type="checkbox" checked={gridConfig.enabled} onChange={(e) => setGridConfig({...gridConfig, enabled: e.target.checked})} className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                  </div>
                  {gridConfig.enabled && (
                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <div>
                        <span className="text-[9px] text-slate-400 font-bold uppercase">Rows</span>
                        <input type="number" min="1" max="4" value={gridConfig.rows} onChange={(e) => setGridConfig({...gridConfig, rows: parseInt(e.target.value)})} className="w-full text-xs p-1 px-2 border rounded bg-white" />
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-400 font-bold uppercase">Cols</span>
                        <input type="number" min="1" max="4" value={gridConfig.cols} onChange={(e) => setGridConfig({...gridConfig, cols: parseInt(e.target.value)})} className="w-full text-xs p-1 px-2 border rounded bg-white" />
                      </div>
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-tighter">Prompt Override</label>
                  <textarea rows={5} value={prompt} onChange={(e) => setPrompt(e.target.value)} className="w-full text-[11px] p-3 border rounded-xl bg-slate-50 focus:bg-white focus:ring-2 focus:ring-blue-100 transition-all" />
                </div>
                <button onClick={handleProcess} disabled={loading || !image} className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all shadow-sm ${loading || !image ? 'bg-slate-200 text-slate-400' : 'bg-blue-600 text-white hover:bg-blue-700'}`}>
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                  {loading ? 'Analyzing...' : 'Parse Figure'}
                </button>
              </div>
            </section>
          ) : (
            <section className="animate-in slide-in-from-left-4">
              <div className="flex items-center gap-2 mb-4 text-blue-600">
                <CropIcon className="w-5 h-5" />
                <h2 className="font-bold uppercase tracking-wider text-xs">Cropping Tools</h2>
              </div>
              <div className="space-y-4">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <label className="block text-[10px] font-bold text-slate-500 mb-1 uppercase tracking-tighter">Find Plot By Topic</label>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={cropQuery} 
                      onChange={(e) => setCropQuery(e.target.value)}
                      placeholder="e.g. ionic conductivity"
                      className="flex-1 text-xs p-2 border rounded bg-white"
                    />
                    <button onClick={handleAutoCrop} disabled={loading || !originalImage} className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-slate-300">
                      <Search className="w-4 h-4" />
                    </button>
                  </div>
                  <p className="text-[9px] text-slate-400 mt-2 italic">Gemini will auto-expand to include labels.</p>
                </div>
                
                {currentBox && (
                  <div className="space-y-2">
                    <button 
                      onClick={() => autoExpand(80)} 
                      className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all border-2 border-blue-100 text-blue-600 hover:bg-blue-50"
                    >
                      <Maximize2 className="w-4 h-4" />
                      Add Safety Padding
                    </button>
                    <button onClick={applyCrop} className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold transition-all shadow-sm bg-blue-600 text-white hover:bg-blue-700">
                      <Check className="w-4 h-4" />
                      Capture Selection
                    </button>
                  </div>
                )}
                <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                  <p className="text-[10px] leading-relaxed text-blue-700">
                    <span className="font-bold">PRO TIP:</span> If the figure is tall, scroll down the workspace to see the full content and adjust the box manually.
                  </p>
                </div>
              </div>
            </section>
          )}
        </aside>

        <section className="flex-1 overflow-hidden relative flex flex-col">
          {error && (
            <div className="m-8 bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-start gap-3 z-50">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div><h4 className="font-bold text-sm">Error</h4><p className="text-xs opacity-80">{error}</p></div>
            </div>
          )}

          {loading && (
            <div className="absolute inset-0 z-50 bg-white/60 backdrop-blur-sm flex items-center justify-center">
              <div className="bg-white p-8 rounded-3xl shadow-xl border flex flex-col items-center max-w-xs text-center animate-in fade-in zoom-in-95">
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin mb-4" />
                <h3 className="font-bold text-slate-800 mb-1">Processing...</h3>
                <p className="text-xs text-slate-500">Communicating with Gemini models for structured figure parsing.</p>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-8">
            {activeTab === 'parser' ? (
              <>
                {result ? (
                  <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <span className="px-2 py-1 bg-blue-50 text-blue-600 text-[10px] font-bold rounded uppercase mb-2 block w-fit tracking-widest">Analysis Result</span>
                          <h2 className="text-2xl font-black text-slate-900 leading-tight">{result.title}</h2>
                        </div>
                        <div className="flex gap-2">
                          <div className="bg-slate-50 p-3 rounded-xl border">
                            <p className="text-[9px] text-slate-400 font-bold uppercase mb-1">X-Axis</p>
                            <p className="text-sm font-bold truncate max-w-[120px]">{result.xAxis.label}</p>
                          </div>
                          <div className="bg-slate-50 p-3 rounded-xl border">
                            <p className="text-[9px] text-slate-400 font-bold uppercase mb-1">Y-Axis</p>
                            <p className="text-sm font-bold truncate max-w-[120px]">{result.yAxis.label}</p>
                          </div>
                        </div>
                      </div>
                      <p className="text-sm text-slate-600 leading-relaxed max-w-3xl">{result.summary}</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col h-[400px]">
                        <div className="flex items-center gap-2 mb-6">
                          <BarChart3 className="w-5 h-5 text-blue-600" />
                          <h3 className="font-bold text-slate-800">Visual Reconstruction</h3>
                        </div>
                        <div className="flex-1 min-h-0">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                              <XAxis type="number" dataKey="xValue" name={result.xAxis.label} unit={result.xAxis.unit} stroke="#94a3b8" fontSize={12} />
                              <YAxis type="number" dataKey="yValue" name={result.yAxis.label} unit={result.yAxis.unit} stroke="#94a3b8" fontSize={12} />
                              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                              <Scatter name="Extracted Data" data={result.dataPoints} fill="#2563eb" />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col h-[400px]">
                        <div className="flex items-center gap-2 mb-6">
                          <LayoutGrid className="w-5 h-5 text-blue-600" />
                          <h3 className="font-bold text-slate-800">Ground Truth Comparison</h3>
                        </div>
                        {comparisonData ? (
                          <div className="flex-1 min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={comparisonData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
                                <YAxis stroke="#94a3b8" fontSize={12} />
                                <Tooltip />
                                <Legend verticalAlign="top" height={36}/>
                                <Line type="monotone" dataKey="extracted" stroke="#2563eb" strokeWidth={3} dot={{ r: 4 }} name="AI Extracted" />
                                <Line type="monotone" dataKey="groundTruth" stroke="#94a3b8" strokeDasharray="5 5" name="Ground Truth" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        ) : (
                          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 space-y-4">
                            <TableIcon className="w-12 h-12 opacity-10" />
                            <p className="text-xs text-center max-w-[200px]">Upload a CSV file to verify AI precision.</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-4 py-20">
                    <div className="bg-white p-12 rounded-3xl shadow-sm border border-slate-200 text-center max-w-md">
                      <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <h3 className="text-xl font-bold text-slate-700">Ready for Parsing</h3>
                      <p className="text-sm leading-relaxed">Ensure your image focus is on the specific plot. If your image contains multiple panels, use the <b>Cropper</b> tab first to isolate the data.</p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="h-full flex flex-col space-y-6 animate-in fade-in min-h-[800px]">
                <div className="bg-white rounded-3xl p-8 shadow-sm border border-slate-200 flex flex-col flex-1 overflow-hidden min-h-0">
                  <div className="flex items-center justify-between w-full mb-6">
                    <div>
                      <h3 className="text-xl font-bold text-slate-800">Interactive Selection Workspace</h3>
                      <p className="text-sm text-slate-500 font-medium">Scroll down for tall images. Drag the box or corners to capture all text.</p>
                    </div>
                  </div>
                  
                  <div 
                    className="relative flex-1 bg-slate-900 rounded-2xl overflow-auto shadow-2xl flex flex-col items-center p-8 border-4 border-slate-800"
                  >
                    {originalImage ? (
                      <div className="relative inline-block">
                        <img 
                          ref={imgRef}
                          src={originalImage} 
                          className="max-w-none block"
                          style={{ maxHeight: 'none' }}
                          alt="Target Figure"
                        />
                        {currentBox && (
                          <div 
                            className="absolute border-4 border-blue-500 bg-blue-500/10 shadow-[0_0_0_9999px_rgba(0,0,0,0.7)] z-20 cursor-move"
                            onMouseDown={(e) => handleMouseDown(e, 'move')}
                            style={{
                              top: `${currentBox.ymin / 10}%`,
                              left: `${currentBox.xmin / 10}%`,
                              width: `${(currentBox.xmax - currentBox.xmin) / 10}%`,
                              height: `${(currentBox.ymax - currentBox.ymin) / 10}%`
                            }}
                          >
                            <div 
                              className="absolute -bottom-4 -right-4 w-10 h-10 bg-blue-600 rounded-full border-4 border-white cursor-nwse-resize flex items-center justify-center shadow-xl z-30"
                              onMouseDown={(e) => handleMouseDown(e, 'resize')}
                            >
                              <Minimize2 className="w-5 h-5 text-white" />
                            </div>
                            
                            <div className="absolute -top-12 left-0 flex gap-2 pointer-events-none">
                              <div className="bg-blue-600 text-white text-[11px] font-bold px-4 py-2 rounded-full shadow-2xl flex items-center gap-2">
                                <Move className="w-3 h-3" />
                                Drag selection
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center p-20">
                        <Upload className="w-16 h-16 mx-auto text-slate-700 mb-4 opacity-50" />
                        <p className="text-slate-500 font-bold">Upload a multi-panel figure to start cropping.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      {image && activeTab === 'parser' && gridConfig.enabled && !result && !loading && (
        <div className="fixed bottom-6 right-6 w-48 bg-white p-2 rounded-xl shadow-2xl border border-blue-100 animate-in slide-in-from-right-4 z-50">
          <p className="text-[9px] font-bold text-blue-600 uppercase mb-2 px-1">Grid Grounding Preview</p>
          <div className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden border">
            <img src={image} className="w-full h-full object-contain opacity-50" />
            <div 
              className="absolute inset-0 grid" 
              style={{ 
                gridTemplateRows: `repeat(${gridConfig.rows}, 1fr)`,
                gridTemplateColumns: `repeat(${gridConfig.cols}, 1fr)`
              }}
            >
              {Array.from({length: gridConfig.rows * gridConfig.cols}).map((_, i) => (
                <div key={i} className="border border-blue-400/30 bg-blue-400/5 flex items-center justify-center">
                   <span className="text-[8px] text-blue-500 font-bold">{i+1}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
