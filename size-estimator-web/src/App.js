import React, { useState } from 'react';
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import awsExports from './aws-exports';
import './App.css';

Amplify.configure(awsExports);

const LAMBDA_BASE_URL = "https://pckg3azoixsoaa6it2oc5q7bbi0ztdex.lambda-url.eu-central-1.on.aws/"; 

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [refClass, setRefClass] = useState('bottle');
  const [refWidth, setRefWidth] = useState(7.0);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile); // Backend expects 'file'
      formData.append('ref_class', refClass);
      formData.append('ref_width', refWidth);

      // Ensure URL ends with /predict
      const url = LAMBDA_BASE_URL.replace(/\/$/, '') + '/predict';

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error uploading file:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Authenticator>
      {({ signOut, user }) => (
        <div className="App">
          <header className="App-header-bar">
            <h1>Size Estimator</h1>
            <div className="user-controls">
              <span>{user?.username}</span>
              <button onClick={signOut} className="sign-out-button">Sign Out</button>
            </div>
          </header>
          
          <main className="App-main">
            <div className="card">
              <h2>Config & Upload</h2>
              
              <div className="config-section">
                <div className="input-group">
                  <label htmlFor="refClass">Reference Object Class:</label>
                  <input 
                    id="refClass"
                    type="text" 
                    value={refClass} 
                    onChange={(e) => setRefClass(e.target.value)}
                    placeholder="e.g., bottle"
                  />
                </div>
                <div className="input-group">
                  <label htmlFor="refWidth">Reference Width (cm):</label>
                  <input 
                    id="refWidth"
                    type="number" 
                    value={refWidth} 
                    onChange={(e) => setRefWidth(e.target.value)}
                    step="0.1"
                  />
                </div>
              </div>

              <div className="upload-section">
                <input 
                  type="file" 
                  onChange={handleFileChange} 
                  accept="image/*" 
                  className="file-input"
                />
                
                {preview && !result && (
                  <div className="image-preview">
                    <p>Preview:</p>
                    <img src={preview} alt="Preview" />
                  </div>
                )}

                <button 
                  onClick={handleUpload} 
                  disabled={!selectedFile || uploading}
                  className="upload-button"
                >
                  {uploading ? 'Processing...' : 'Upload & Estimate'}
                </button>
              </div>
            </div>

            {result && (
              <div className="card result-section">
                <h2>Result</h2>
                {result.image_base64 && (
                  <div className="result-image">
                    <img 
                      src={`data:image/jpeg;base64,${result.image_base64}`} 
                      alt="Processed Result" 
                    />
                  </div>
                )}
                
                <div className="result-details">
                  <h3>Detections:</h3>
                  {result.objects && result.objects.length > 0 ? (
                    <ul>
                      {result.objects.map((obj, index) => (
                        <li key={index}>
                          <strong>{obj.label}</strong>: {obj.estimated_width_cm ? `${obj.estimated_width_cm} cm` : 'Width unknown'}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No objects detected.</p>
                  )}
                  
                  {!result.reference_found && (
                    <p className="warning">Reference object ({refClass}) not found. Sizes cannot be estimated.</p>
                  )}
                </div>
              </div>
            )}
          </main>
        </div>
      )}
    </Authenticator>
  );
}

export default App;
