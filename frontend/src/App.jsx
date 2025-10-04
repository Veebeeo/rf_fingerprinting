import { useState, useRef } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const AlertIcon = ({ colorClass }) => (
    <svg className={`w-6 h-6 mr-2 ${colorClass}`} fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.22 3.006-1.742 3.006H4.42c-1.522 0-2.492-1.672-1.742-3.006l5.58-9.92zM10 13a1 1 0 110-2 1 1 0 010 2zm-1-4a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
    </svg>
);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.npy')) {
      setSelectedFile(file);
      setError(null);
      setAnalysisResult(null);
    } else {
      setSelectedFile(null);
      setError("Invalid file type. Please upload a .npy file.");
    }
  };
  
  const handleSampleFile = async (fileUrl, sampleFileName) => {
    setIsLoading(true);
    setAnalysisResult(null);
    setError(null);
    try {
      const response = await fetch(fileUrl);
      if (!response.ok) {
        throw new Error(`Network response was not ok for ${sampleFileName}`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], sampleFileName, { type: "application/octet-stream" });
      
      setSelectedFile(sampleFile);

    } catch (err) {
      setError("Failed to load sample file. Please check the URL.");
      console.error(err);
    } finally {
        setIsLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) { setError("Please select a file."); return; }
    setIsLoading(true);
    setAnalysisResult(null);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const url = `https://spectrum-intelligence.onrender.com/predict/`;
      const response = await axios.post(url, formData);
      setAnalysisResult(response.data);
    } catch (err) {
      setError("Analysis failed. Backend might be offline or the file is invalid.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const getResultCardStyle = () => {
    if (!analysisResult) return 'border-gray-300';
    if (analysisResult.is_anomaly) return 'border-yellow-400 bg-yellow-50';
    return 'border-gray-300 bg-white';
  };
  const getResultTextStyle = () => {
    if (!analysisResult) return 'text-gray-800';
    if (analysisResult.is_anomaly) return 'text-yellow-800';
    return 'text-ocean';
  };

  return (
    <div className="bg-dreamy font-sans min-h-screen">
      <header className="bg-ocean text-white shadow-lg">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-3xl font-bold leading-tight">Spectrum Intelligence</h1>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto py-10 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg-col-span-1 flex flex-col space-y-8">
          <div className="bg-white p-8 rounded-lg shadow-md border border-gray-200">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Signal Upload</h2>
            <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-56 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
              <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
                <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                <p className="text-xs text-gray-400">RF signal file (.npy)</p>
                {selectedFile && <p className="text-sm text-ocean mt-4 font-semibold">{selectedFile.name}</p>}
              </div>
              <input ref={fileInputRef} id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept=".npy" />
            </label>
            <div className="mt-6">
              <button onClick={handleAnalyze} disabled={isLoading || !selectedFile} className="w-full bg-ocean hover:opacity-90 disabled:bg-gray-400 text-white rounded-lg px-4 py-3 font-bold transition-colors">
                {isLoading ? "Analyzing..." : "Analyze Signal"}
              </button>
            </div>
          </div>

          <div className="bg-white p-8 rounded-lg shadow-md border border-gray-200">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Test with a Fast Demo Sample</h2>
            <div className="flex flex-col space-y-4">
                 <button 
                    onClick={() => handleSampleFile('https://media.githubusercontent.com/media/Veebeeo/rf_fingerprinting/main/frontend/public/samp_demo_1_stealth_incursion.npy?download=true', 'samp_demo_1_stealth_incursion.npy')} 
                    disabled={isLoading} 
                    className="w-full bg-ocean hover:opacity-90 disabled:bg-gray-400 text-white rounded-lg px-4 py-3 font-bold transition-colors">
                    Load Stealth Demo
                </button>
                 <button 
                    onClick={() => handleSampleFile('https://media.githubusercontent.com/media/Veebeeo/rf_fingerprinting/main/frontend/public/samp_demo_2_disruption_attack.npy?download=true', 'samp_demo_2_disruption_attack.npy')} 
                    disabled={isLoading} 
                    className="w-full bg-ocean hover:opacity-90 disabled:bg-gray-400 text-white rounded-lg px-4 py-3 font-bold transition-colors">
                    Load Disruption Demo
                </button>
                <button 
                    onClick={() => handleSampleFile('https://media.githubusercontent.com/media/Veebeeo/rf_fingerprinting/main/frontend/public/samp_demo_real_world_QPSK_test.npy?download=true', 'samp_demo_real_world_QPSK_test.npy')} 
                    disabled={isLoading} 
                    className="w-full bg-ocean hover:opacity-90 disabled:bg-gray-400 text-white rounded-lg px-4 py-3 font-bold transition-colors">
                    Load QPSK Demo
                </button>
            </div>
          </div>
        </div>

        <div className="lg-col-span-2 flex flex-col space-y-8">
          {error && ( <div className="p-4 text-sm text-red-800 bg-red-100 border border-red-200 rounded-lg"> <strong>Error:</strong>&nbsp;{error} </div> )}
          {analysisResult && (
            <div className={`p-6 border ${getResultCardStyle()} rounded-lg shadow-md`}>
              <h3 className={`text-xl font-bold mb-4 flex items-center ${getResultTextStyle()}`}> <AlertIcon colorClass={getResultTextStyle()} /> Analysis Complete </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center text-lg"> <span className="font-medium text-gray-600">Classification:</span> <span className={`font-bold px-3 py-1 rounded-full text-sm ${getResultTextStyle()} ${getResultCardStyle().split(' ')[1]}`}> {analysisResult.predicted_device.toUpperCase()} </span> </div>
                <div className="flex justify-between items-center text-lg"> <span className="font-medium text-gray-600">Confidence:</span> <span className={`font-mono font-semibold ${getResultTextStyle()}`}> {(analysisResult.confidence_score * 100).toFixed(4)}% </span> </div>
                <div className="flex justify-between items-center text-lg"> <span className="font-medium text-gray-600">Anomaly Detected:</span> <span className={`font-mono font-semibold ${getResultTextStyle()}`}> {analysisResult.is_anomaly ? "YES" : "NO"} </span> </div>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
            <h3 className="font-bold text-gray-800 text-lg mb-2">Reconstruction Error Analysis</h3>
            <p className="text-sm text-gray-500 mb-4">The yellow bars highlight areas where the AI's reconstruction differed from the original signal, indicating anomalous or important features.</p>
            {analysisResult && analysisResult.reconstruction_error_map && analysisResult.signal_magnitude ? (
              <Plot
                data={[
                  {
                    x: Array.from(Array(analysisResult.signal_magnitude.length).keys()),
                    y: analysisResult.signal_magnitude,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#006989', width: 2 },
                    name: 'Signal Magnitude'
                  },
                  {
                    x: Array.from(Array(analysisResult.reconstruction_error_map.length).keys()),
                    y: analysisResult.reconstruction_error_map.map(s => s * Math.max(...analysisResult.signal_magnitude)),
                    type: 'bar',
                    marker: { color: '#FBBF24', opacity: 0.5 },
                    name: 'Reconstruction Error'
                  }
                ]}
                layout={{
                  barmode: 'overlay',
                  showlegend: false,
                  autosize: true,
                  margin: { l: 40, r: 20, b: 40, t: 20 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: '#f9fafb',
                  font: { color: '#374151' }
                }}
                useResizeHandler={true}
                className="w-full h-64"
              />
            ) : (
              <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed"><p className="text-gray-400">Analyze a signal to see the results.</p></div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;