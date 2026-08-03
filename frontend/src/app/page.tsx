"use client";

import { useState } from "react";
import axios from "axios";
import { Upload, Cpu, Activity, ShieldAlert, CheckCircle, RefreshCw } from "lucide-react";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  // Submit image to FastAPI Backend
  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Calling our FastAPI backend running on Google Colab via Ngrok
      const response = await axios.post("ttps://neon-jolt-evil.ngrok-free.dev/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true", // <--- Yeh line add kar di hai
        },
      });
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError("Failed to connect to backend server. Ensure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 font-sans">
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-10 border-b border-slate-800 pb-6 flex flex-col md:flex-row justify-between items-center">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-emerald-400 flex items-center gap-2">
            🌱 PlantDoc XAI Dashboard
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Explainable AI Microservice for Automated Plant Disease Detection & System Benchmarking
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex gap-3">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Backend Online
          </span>
        </div>
      </header>

      {/* Main Container */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Upload & Preview */}
        <div className="lg:col-span-1 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-emerald-400" /> Upload Plant Leaf
            </h2>
            <p className="text-sm text-slate-400 mb-6">
              Upload an image of a crop leaf to analyze disease symptoms using MobileNetV3 and Grad-CAM.
            </p>

            {/* Dropzone Box */}
            <label className="border-2 border-dashed border-slate-700 hover:border-emerald-500 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer bg-slate-950/50 transition-all group mb-6">
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="w-full h-48 object-cover rounded-lg shadow-md" />
              ) : (
                <div className="text-center py-6">
                  <Upload className="w-10 h-10 text-slate-500 group-hover:text-emerald-400 mx-auto mb-3 transition-colors" />
                  <span className="text-sm text-slate-300 font-medium">Click to browse image</span>
                  <p className="text-xs text-slate-500 mt-1">Supports JPG, PNG, JPEG</p>
                </div>
              )}
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
            </label>
          </div>

          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className={`w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2 shadow-lg transition-all ${
              !selectedFile || loading
                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                : "bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer"
            }`}
          >
            {loading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" /> Processing XAI...
              </>
            ) : (
              <>Analyze Plant Health</>
            )}
          </button>

          {error && <p className="text-red-400 text-xs mt-3 text-center">{error}</p>}
        </div>

        {/* Right Column: Results & XAI Heatmaps */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Prediction Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-emerald-400" /> Diagnostic Results
            </h2>

            {result ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 uppercase tracking-wider">Predicted Class ID</span>
                  <p className="text-3xl font-black text-emerald-400 mt-1">{result.prediction}</p>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 uppercase tracking-wider">Confidence Score</span>
                  <p className="text-3xl font-black text-indigo-400 mt-1">
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-10 bg-slate-950/40 rounded-xl border border-slate-800/50">
                <p className="text-slate-500 text-sm">Upload an image and click analyze to view diagnostics.</p>
              </div>
            )}
          </div>

          {/* System Benchmarks Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-emerald-400" /> System Performance Benchmarks
            </h2>

            {result && result.benchmarks ? (
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-center">
                  <span className="text-xs text-slate-400 block">Inference Latency</span>
                  <span className="text-lg font-bold text-amber-400">{result.benchmarks.inference_time_ms.toFixed(2)} ms</span>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-center">
                  <span className="text-xs text-slate-400 block">RAM Consumption</span>
                  <span className="text-lg font-bold text-sky-400">{result.benchmarks.ram_usage_mb.toFixed(1)} MB</span>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-center">
                  <span className="text-xs text-slate-400 block">CPU Utilization</span>
                  <span className="text-lg font-bold text-emerald-400">{result.benchmarks.cpu_percent}%</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-6 bg-slate-950/40 rounded-xl border border-slate-800/50">
                <p className="text-slate-500 text-sm">Benchmarks will populate here after execution.</p>
              </div>
            )}
          </div>

          {/* XAI Heatmaps Grid */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" /> Explainable AI (XAI) Visualizations
            </h2>

            {result && result.heatmaps ? (
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(result.heatmaps).map(([key, value]: [string, any]) => (
                  <div key={key} className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col items-center">
                    <span className="text-xs font-semibold uppercase text-slate-400 mb-2">{key}</span>
                    <img
                      src={`data:image/png;base64,${value}`}
                      alt={key}
                      className="w-full h-40 object-contain rounded-lg bg-black"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-10 bg-slate-950/40 rounded-xl border border-slate-800/50">
                <p className="text-slate-500 text-sm">XAI heatmaps will render here once processed.</p>
              </div>
            )}
          </div>

        </div>

      </div>
    </main>
  );
}