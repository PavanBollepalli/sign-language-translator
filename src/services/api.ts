const API_BASE_URL = 'http://localhost:5000/api';

export interface DetectionResult {
  letter: string;
  confidence: number;
  class_idx: number;
}

export interface ApiHealthResponse {
  status: string;
  model_loaded: boolean;
}

/**
 * Check if the API is available and if the model is loaded
 */
export async function checkApiHealth(): Promise<ApiHealthResponse> {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) {
      throw new Error(`API Health check failed: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
}

/**
 * Detect sign language from an image
 * @param imageData Base64 encoded image data
 */
export async function detectSign(imageData: string): Promise<DetectionResult> {
  try {
    const response = await fetch('/api/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error detecting sign:', error);
    throw error;
  }
}

/**
 * Translate text to sign language representations
 * @param text Text to translate to sign language
 */
export async function translateText(text: string): Promise<{ text: string; signs: string[] }> {
  try {
    const response = await fetch('/api/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error translating text:', error);
    throw error;
  }
}
