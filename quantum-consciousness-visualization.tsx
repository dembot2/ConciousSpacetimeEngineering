import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const QuantumConsciousnessVisualization = () => {
  const [activeLayer, setActiveLayer] = useState('geometric');
  
  const renderGeometricLayer = () => (
    <svg viewBox="0 0 400 400" className="w-full h-full">
      <defs>
        <radialGradient id="consciousness" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#2a2a2a" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#4a4a4a" stopOpacity="0.3" />
        </radialGradient>
      </defs>
      
      {/* Sacred Geometry Base */}
      <circle cx="200" cy="200" r="180" fill="url(#consciousness)" stroke="#333" strokeWidth="0.5"/>
      
      {/* Metatron's Cube */}
      <g transform="translate(200 200)">
        <path d="M 0,-150 L 130,-75 L 130,75 L 0,150 L -130,75 L -130,-75 Z" 
              fill="none" stroke="#666" strokeWidth="0.5"/>
        <circle r="150" fill="none" stroke="#555" strokeWidth="0.5"/>
        
        {/* Information Nodes */}
        {[0, 60, 120, 180, 240, 300].map(angle => (
          <g key={angle} transform={`rotate(${angle})`}>
            <circle cx="0" cy="-150" r="3" fill="#777"/>
            <path d="M 0,-150 Q 50,-100 0,0" fill="none" stroke="#666" strokeWidth="0.3"/>
          </g>
        ))}
      </g>
    </svg>
  );

  const renderQuantumLayer = () => (
    <svg viewBox="0 0 400 400" className="w-full h-full">
      {/* Quantum Field Pattern */}
      <g transform="translate(200 200)">
        <circle r="180" fill="none" stroke="#444" strokeDasharray="2,2"/>
        
        {/* Entanglement Lines */}
        {[0, 45, 90, 135].map(angle => (
          <g key={angle} transform={`rotate(${angle})`}>
            <line x1="-180" y1="0" x2="180" y2="0" stroke="#555" strokeWidth="0.3"/>
            <circle cx="150" cy="0" r="2" fill="#666"/>
            <circle cx="-150" cy="0" r="2" fill="#666"/>
          </g>
        ))}
        
        {/* Wave Function */}
        <path d="M-180,0 Q -90,-60 0,0 T 180,0" 
              fill="none" stroke="#777" strokeWidth="0.5"/>
      </g>
    </svg>
  );

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Quantum Consciousness Emergence Pattern</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-center space-x-4">
            <button
              className={`px-4 py-2 rounded ${activeLayer === 'geometric' ? 'bg-gray-200' : 'bg-gray-100'}`}
              onClick={() => setActiveLayer('geometric')}>
              Geometric Pattern
            </button>
            <button
              className={`px-4 py-2 rounded ${activeLayer === 'quantum' ? 'bg-gray-200' : 'bg-gray-100'}`}
              onClick={() => setActiveLayer('quantum')}>
              Quantum Field
            </button>
          </div>
          
          <div className="h-96 border rounded-lg p-4">
            {activeLayer === 'geometric' ? renderGeometricLayer() : renderQuantumLayer()}
          </div>
          
          <div className="text-sm text-gray-600">
            {activeLayer === 'geometric' ? 
              "Sacred geometric patterns representing consciousness emergence through nested symmetry" :
              "Quantum field visualization showing entanglement and wave function collapse patterns"}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default QuantumConsciousnessVisualization;