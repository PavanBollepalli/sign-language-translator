import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import { Camera, Volume2, VolumeX, Loader2, Trash2, FlipHorizontal } from 'lucide-react';
import { detectSign, checkApiHealth, DetectionResult } from '../services/api';

const TranslatePage = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const [model, setModel] = useState<handpose.HandPose | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isApiAvailable, setIsApiAvailable] = useState(false);
  const [isProcessingFrame, setIsProcessingFrame] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [detectionHistory, setDetectionHistory] = useState<string[]>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.7);
  const [lastDetectedLetter, setLastDetectedLetter] = useState<string | null>(null);
  const [stableDetectionCount, setStableDetectionCount] = useState<number>(0);
  const [isStabilizing, setIsStabilizing] = useState<boolean>(false);
  const [isMirrored, setIsMirrored] = useState<boolean>(true);
  
  // Check if the backend API is available
  /*Start the backend API server:
  python backend/api.py
  Start the Frontend React Application:
  npm install
  npm run dev
  Make sure your trained model is available to the backend
  python backend/train.py --data_dir D:\Projects\HandShake\asl_dataset
  */
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const health = await checkApiHealth();
        setIsApiAvailable(health.status === 'ok' && health.model_loaded);
        console.log('API health check:', health);
        
        if (health.model_loaded) {
          // If backend model is loaded, we don't need the TF.js handpose model
          setIsModelLoading(false);
        }
      } catch (error) {
        console.error('API not available:', error);
        setIsApiAvailable(false);
      }
    };
    
    checkApiStatus();
  }, []);
  
  // Load TF handpose model only if API is not available
  useEffect(() => {
    const loadModel = async () => {
      if (!isApiAvailable) {
        try {
          await tf.ready();
          await tf.setBackend('webgl');
          
          const loadedModel = await handpose.load();
          setModel(loadedModel);
          console.log('Handpose model loaded');
        } catch (error) {
          console.error('Error loading model:', error);
        }
      }
      setIsModelLoading(false);
    };
    
    if (isModelLoading && !isApiAvailable) {
      loadModel();
    }
  }, [isApiAvailable, isModelLoading]);

  // Clear detection history
  const clearHistory = () => {
    setDetectionHistory([]);
  };

  // Apply image preprocessing before sending to API
  const preprocessImage = useCallback((imageSrc: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        // Create a canvas to manipulate the image
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        
        if (ctx) {
          // Draw image with higher contrast and brightness
          ctx.filter = 'contrast(1.2) brightness(1.1)';
          ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);
          
          // Return processed image
          resolve(canvas.toDataURL('image/jpeg', 0.95));
        } else {
          resolve(imageSrc); // Fall back to original if context not available
        }
      };
      img.src = imageSrc;
    });
  }, []);

  // Memoize the captureFrame function with useCallback
  // Function to speak detected letters - defined below
  
  const captureFrame = useCallback(async () => {
    if (
      webcamRef.current &&
      webcamRef.current.video &&
      webcamRef.current.video.readyState === 4 &&
      !isProcessingFrame
    ) {
      try {
        setIsProcessingFrame(true);
        
        // Capture frame as base64 image
        const rawImageSrc = webcamRef.current.getScreenshot();
        
        if (rawImageSrc && isApiAvailable) {
          // Preprocess image before sending to API
          const processedImageSrc = await preprocessImage(rawImageSrc);
          
          // Send to backend API
          const result = await detectSign(processedImageSrc);
          setDetectionResult(result);
          
          // Only accept predictions with confidence above threshold
          if (result.confidence > confidenceThreshold) {
            const letter = result.letter.toString();
            
            // Stabilization logic
            if (letter === lastDetectedLetter) {
              // Same letter detected consecutively
              setStableDetectionCount(prev => prev + 1);
              
              // After 3 consistent detections, add to history
              if (stableDetectionCount >= 2 && !isStabilizing) {
                setIsStabilizing(true);
                
                // Add to history if it's different from the last added letter
                if (detectionHistory.length === 0 || detectionHistory[detectionHistory.length - 1] !== letter) {
                  setDetectionHistory(prev => [...prev.slice(-29), letter]); // Keep last 30 detections
                  if (isSpeechEnabled) {
                    speak(letter);
                  }
                }
                
                // Reset stabilizing flag after a delay to prevent rapid additions
                setTimeout(() => {
                  setIsStabilizing(false);
                }, 1000);
              }
            } else {
              // Different letter detected, reset counter
              setStableDetectionCount(0);
              setLastDetectedLetter(letter);
            }
          } else {
            // Low confidence, increment stability counter for "Unknown"
            if (lastDetectedLetter === "Unknown") {
              setStableDetectionCount(prev => prev + 1);
            } else {
              setStableDetectionCount(0);
              setLastDetectedLetter("Unknown");
            }
          }
        } else if (model) {
          // Use TF.js handpose model as fallback
          const video = webcamRef.current.video;
          const predictions = await model.estimateHands(video);
          
          if (predictions.length > 0) {
            if (isSpeechEnabled) {
              speak('Hand detected!');
            }
          }
        }
      } catch (error) {
        console.error('Error processing frame:', error);
      } finally {
        setIsProcessingFrame(false);
      }
    }
  }, [
    webcamRef,
    isProcessingFrame,
    isApiAvailable,
    preprocessImage,
    confidenceThreshold,
    lastDetectedLetter,
    stableDetectionCount,
    isStabilizing,
    detectionHistory,
    isSpeechEnabled,
    model
  ]);

  // Function to speak detected letters
  const speak = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isCameraOn && !isModelLoading) {
      interval = setInterval(() => {
        captureFrame();
      }, 500); // Capture every 500ms to avoid overwhelming the API
    }
    return () => clearInterval(interval);
  }, [captureFrame, isCameraOn, isModelLoading]);

  const toggleCamera = () => {
    setIsCameraOn(!isCameraOn);
  };

  // Add confidence threshold slider
  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfidenceThreshold(parseFloat(e.target.value));
  };

  // Toggle camera mirroring
  const toggleMirroring = () => {
    setIsMirrored(!isMirrored);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Sign Language Translator</h1>
          <p className="text-gray-400">
            {isApiAvailable 
              ? "Using trained ASL detection model" 
              : "Using browser-based hand detection"
            }
          </p>
        </header>

        <div className="max-w-3xl mx-auto">
          <div className="relative bg-gray-800 rounded-lg overflow-hidden shadow-xl">
            {isModelLoading ? (
              <div className="flex items-center justify-center h-[480px]">
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-6 h-6 animate-spin" />
                  <span>Loading AI model...</span>
                </div>
              </div>
            ) : isCameraOn ? (
              <Webcam
                ref={webcamRef}
                audio={false}
                className="w-full"
                screenshotFormat="image/jpeg"
                mirrored={isMirrored}
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: "user",
                }}
              />
            ) : (
              <div className="flex items-center justify-center h-[480px] bg-gray-700">
                <p className="text-gray-400">Camera is turned off</p>
              </div>
            )}
            
            <div className="absolute bottom-4 right-4 flex space-x-2">
              <button
                className="p-2 bg-gray-700 rounded-full hover:bg-gray-600 transition-colors"
                onClick={toggleMirroring}
                title={isMirrored ? "Disable mirroring" : "Enable mirroring"}
              >
                <FlipHorizontal className="w-6 h-6" />
              </button>
              <button
                className="p-2 bg-gray-700 rounded-full hover:bg-gray-600 transition-colors"
                onClick={() => setIsSpeechEnabled(!isSpeechEnabled)}
                title={isSpeechEnabled ? "Disable speech" : "Enable speech"}
              >
                {isSpeechEnabled ? (
                  <Volume2 className="w-6 h-6" />
                ) : (
                  <VolumeX className="w-6 h-6" />
                )}
              </button>
              <button 
                className="p-2 bg-gray-700 rounded-full hover:bg-gray-600 transition-colors"
                onClick={toggleCamera}
                title={isCameraOn ? "Turn camera off" : "Turn camera on"}
              >
                <Camera className="w-6 h-6" />
              </button>
            </div>
          </div>

          <div className="mt-6 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Translation</h2>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="min-h-[40px]">
                <p className="text-xl">
                  {detectionResult ? 
                    `Detected: ${detectionResult.letter} (Confidence: ${(detectionResult.confidence * 100).toFixed(1)}%)` : 
                    'No signs detected yet...'
                  }
                </p>
              </div>
              {detectionHistory.length > 0 && (
                <div className="mt-4">
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="text-lg font-medium">Detection History:</h3>
                    <button 
                      className="p-2 bg-gray-600 rounded-md hover:bg-gray-500 transition-colors flex items-center"
                      onClick={clearHistory}
                    >
                      <Trash2 className="w-4 h-4 mr-1" />
                      Clear
                    </button>
                  </div>
                  <p className="text-2xl tracking-wider">
                    {detectionHistory.join('')}
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block mb-2 text-sm font-medium">
                  Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={handleThresholdChange}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <p className="mt-1 text-xs text-gray-400">Higher values provide more accurate but fewer predictions</p>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="mirror-toggle"
                  checked={isMirrored}
                  onChange={toggleMirroring}
                  className="rounded bg-gray-700"
                />
                <label htmlFor="mirror-toggle" className="text-sm">
                  Mirror webcam view (set to OFF for correct sign language orientation)
                </label>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Instructions</h2>
            <ul className="list-disc list-inside space-y-2 text-gray-300">
              <li>Position your hand clearly in front of the camera</li>
              <li>Make sure there's good lighting</li>
              <li>Keep your hand steady while signing</li>
              <li>Try to sign against a plain background</li>
              <li>Hold each sign until it's recognized consistently</li>
              <li>Adjust the confidence threshold if needed</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslatePage;