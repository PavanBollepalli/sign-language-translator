import React from 'react';
import { Link } from 'react-router-dom';
import { Hand as Hands, Heart, Globe2, Brain } from 'lucide-react';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-purple-900 to-gray-900">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-white mb-6 animate-fade-in">
            Breaking Barriers Through
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600">
              {" "}Sign Language AI
            </span>
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Transform the way you communicate with our cutting-edge AI-powered sign language translator.
            Real-time translation that brings people together.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800 bg-opacity-50 p-8 rounded-xl backdrop-blur-sm">
            <Hands className="w-12 h-12 text-purple-400 mb-4" />
            <h3 className="text-2xl font-semibold text-white mb-4">Real-Time Translation</h3>
            <p className="text-gray-300">
              Advanced AI technology translates sign language to text and speech instantly,
              making communication seamless and natural.
            </p>
          </div>

          <div className="bg-gray-800 bg-opacity-50 p-8 rounded-xl backdrop-blur-sm">
            <Brain className="w-12 h-12 text-purple-400 mb-4" />
            <h3 className="text-2xl font-semibold text-white mb-4">AI-Powered Accuracy</h3>
            <p className="text-gray-300">
              Our sophisticated machine learning models ensure high accuracy in gesture
              recognition and translation.
            </p>
          </div>

          <div className="bg-gray-800 bg-opacity-50 p-8 rounded-xl backdrop-blur-sm">
            <Globe2 className="w-12 h-12 text-purple-400 mb-4" />
            <h3 className="text-2xl font-semibold text-white mb-4">Universal Access</h3>
            <p className="text-gray-300">
              Break down communication barriers and connect with anyone, anywhere,
              regardless of hearing ability.
            </p>
          </div>
        </div>

        <div className="text-center">
          <Link
            to="/translate"
            className="inline-flex items-center px-8 py-4 text-lg font-semibold text-white bg-gradient-to-r from-purple-500 to-pink-500 rounded-full hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105"
          >
            Start Translating Now
            <Heart className="ml-2 w-5 h-5" />
          </Link>
        </div>

        <div className="mt-24">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <img
                src="https://images.unsplash.com/photo-1531936402304-e49b12867cf1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1000&q=80"
                alt="People communicating in sign language"
                className="rounded-xl shadow-2xl"
              />
            </div>
            <div>
              <h2 className="text-4xl font-bold text-white mb-6">
                Empowering Communication for Everyone
              </h2>
              <p className="text-gray-300 text-lg mb-8">
                Our mission is to make communication accessible to everyone. With our
                AI-powered sign language translator, we're breaking down barriers and
                creating a more inclusive world.
              </p>
              <ul className="space-y-4 text-gray-300">
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-purple-400 rounded-full mr-3"></span>
                  Real-time translation capabilities
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-purple-400 rounded-full mr-3"></span>
                  High accuracy and reliability
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-purple-400 rounded-full mr-3"></span>
                  Easy to use interface
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;