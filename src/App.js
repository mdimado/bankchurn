import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [emotions, setEmotions] = useState([]);

  const handleImageUpload = async (event) => {
    const formData = new FormData();
    formData.append('image', event.target.files[0]);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setEmotions(response.data.emotions);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <div>
        {emotions.map((emotion, index) => (
          <div key={index}>{emotion}</div>
        ))}
      </div>
    </div>
  );
}

export default App;
