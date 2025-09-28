import { useState, useRef } from 'react';
import axios from 'axios';

// Simple icon components for a better UI
const UploadIcon = () => (
  <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
  </svg>
);

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
      setError(null); // Clear previous errors
    } else {
      setSelectedFile(null);
      setError("Invalid file type. Please upload a .npy file.");
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    setAnalysisResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setAnalysisResult(response.data);
    } catch (err) {
      setError("An error occurred during analysis. Is the backend server running?");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const getResultCardStyle = () => {
    if (!analysisResult) return 'border-gray-300';
    if (analysisResult.predicted_device === 'hostile_drone') return 'border-red-500 bg-red-900 bg-opacity-20';
    if (analysisResult.predicted_device === 'friendly_drone') return 'border-green-500 bg-green-900 bg-opacity-20';
    return 'border-blue-500 bg-blue-900 bg-opacity-20';
  };

  const getResultTextStyle = () => {
    if (!analysisResult) return 'text-gray-200';
    if (analysisResult.predicted_device === 'hostile_drone') return 'text-red-400';
    if (analysisResult.predicted_device === 'friendly_drone') return 'text-green-400';
    return 'text-blue-400';
  };


  return (
    <div className="bg-gray-900 min-h-screen flex flex-col items-center justify-center font-sans p-4 text-white">
      <div className="w-full max-w-2xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300">
            Project AETHER
          </h1>
          <p className="text-lg text-gray-400 mt-2">AI-Enhanced Threat Hunting in Electromagnetic Realms</p>
        </header>

        <main className="bg-gray-800 rounded-xl shadow-2xl p-6 md:p-8 border border-gray-700">
          <div 
            className="flex items-center justify-center w-full"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              fileInputRef.current.files = e.dataTransfer.files;
              handleFileChange({ target: { files: e.dataTransfer.files } });
            }}
          >
            <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <UploadIcon />
                <p className="mb-2 text-sm text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                <p className="text-xs text-gray-500">A single RF signal file (.npy)</p>
                {selectedFile && <p className="text-sm text-blue-400 mt-4">{selectedFile.name}</p>}
              </div>
              <input ref={fileInputRef} id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept=".npy" />
            </label>
          </div>

          <div className="mt-6">
            <button
              onClick={handleSubmit}
              disabled={isLoading || !selectedFile}
              className="w-full text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed focus:ring-4 focus:outline-none focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-3 text-center transition-colors"
            >
              {isLoading ? 'Analyzing Spectrum...' : 'Analyze Signal Fingerprint'}
            </button>
          </div>

          {error && (
            <div className="mt-6 p-4 text-sm text-red-300 bg-red-800 bg-opacity-30 rounded-lg flex items-center">
                <AlertIcon colorClass="text-red-400"/>
                <strong>Error:</strong>&nbsp;{error}
            </div>
          )}

          {analysisResult && (
            <div className={`mt-6 p-5 border-2 ${getResultCardStyle()} rounded-lg shadow-lg`}>
              <h3 className={`text-xl font-bold mb-4 flex items-center ${getResultTextStyle()}`}>
                <AlertIcon colorClass={getResultTextStyle()} />
                Analysis Complete
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center text-lg">
                  <span className="font-medium text-gray-300">Predicted Device:</span>
                  <span className={`font-bold px-3 py-1 rounded-full ${getResultTextStyle()} ${getResultCardStyle().split(' ')[1]}`}>
                    {analysisResult.predicted_device.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between items-center text-lg">
                  <span className="font-medium text-gray-300">Confidence Score:</span>
                  <span className={`font-mono text-lg font-semibold ${getResultTextStyle()}`}>
                    {(analysisResult.confidence_score * 100).toFixed(4)}%
                  </span>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-600">
                 <h4 className="text-md font-semibold text-gray-400 mb-2">Full Probability Distribution:</h4>
                 <ul className="space-y-1 text-sm font-mono">
                    {Object.entries(analysisResult.details).map(([device, score]) => (
                        <li key={device} className="flex justify-between items-center">
                            <span className="text-gray-400">{device}:</span>
                            <span className="text-gray-300">{(score * 100).toExponential(4)}%</span>
                        </li>
                    ))}
                 </ul>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;