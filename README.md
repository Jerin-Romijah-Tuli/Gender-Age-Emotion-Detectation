
# Age, Gender, and Emotion Detection Using AI

![image alt](https://github.com/Jerin-Romijah-Tuli/Gender-Age-Emotion-Detectation/blob/514b57abafa210cc5bde1bf3faa2ebfa5e5843dc/image%20(1).jpg)  
*Unlock the power of AI to detect age, gender, and emotions in real-time or from images!*

---

## Table of Contents
- [What’s This Project About?](#whats-this-project-about)
- [Awesome Features](#awesome-features)
  - [Emotion Detection](#emotion-detection)
- [Tech Stack](#tech-stack)
- [Get Started](#get-started)
- [How to Use It](#how-to-use-it)
  - [Live Webcam Magic](#live-webcam-magic)
  - [Upload and Detect](#upload-and-detect)
- [Contributing](#contributing)
- [License](#license)
- [About the Developer](#about-the-developer)

---

## What’s This Project About?
This project uses cutting-edge deep learning to analyze faces and predict **age**, **gender**, and **emotions**. Whether you’re streaming live from your webcam or uploading a photo, the sleek Gradio interface makes it fun and easy to explore these predictions—complete with confidence scores and lightning-fast processing times! The emotion detection component leverages a convolutional neural network (CNN) trained on the [Kaggle Facial Expression Recognition Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) to identify emotions like Angry, Happy, Neutral, Sad, and Surprise.

---

## Awesome Features
Here’s what makes this project stand out:

- **Real-Time Webcam Detection**  
  Fire up your webcam and watch the AI detect faces, guessing age, gender, and emotions on the fly. It’s like magic, but with science!

- **Image Upload Detection**  
  Drop in a photo, hit "Detect," and see the results pop up—complete with a labeled image and detailed stats.

- **Confidence Scores**  
  Every prediction comes with a confidence percentage, so you know how sure the AI is. No guesswork here!

- **Processing Time**  
  Curious how fast it works? Get the exact time taken for each detection, displayed in both the UI and terminal.

- **Interactive Gradio UI**  
  A slick, easy-to-use web interface that’s as fun to play with as it is powerful.

- **Terminal Insights**  
  Detailed logs print to your terminal—perfect for debugging or just geeking out over the results.

- **Start/Stop Webcam Controls**  
  Take charge with buttons to start or stop the webcam stream whenever you want. Pause it, and the last frame stays on screen!

- **Persistent Results**  
  Stop the webcam, and the final frame plus its predictions stick around in the UI—no disappearing acts.

### Emotion Detection
- **Real-Time Emotion Detection**: Detect emotions from live webcam feeds.
- **Customizable Classes**: Modify the number of emotion classes based on your requirements.
- **Pre-Trained Models**: Experiment with different pre-trained models for improved accuracy.
- **Easy Integration**: Use OpenCV for face detection and TensorFlow/Keras for emotion classification.
- **Interactive Testing**: Run the `test.py` script to see emotion detection in action.

---

## Tech Stack
- **Python 3.x**: The backbone of the project.
- **OpenCV**: Handles all the face-finding and image processing.
- **Keras/TensorFlow**: Powers the deep learning models.
- **Gradio**: Brings the interactive web UI to life.
- **Caffe Models**: Pre-trained for age and gender detection.
- **Custom Keras Model**: Trained for emotion detection.

---

## Get Started
Ready to dive in? Here’s how to set it up:

1. **Clone the Repo**  
   ```bash
   git clone https://github.com/yourusername/age-gender-emotion-detection.git
   cd age-gender-emotion-detection
   ```

2. **Install the Goodies**  
   ```bash
   pip install -r requirements.txt
   ```
   *Make sure you’re rocking Python 3.x (version 3.10 or higher recommended)!*

3. **Grab the Models**  
   Drop these files into your project folder:  
   - `opencv_face_detector.pbtxt`  
   - `opencv_face_detector_uint8.pb`  
   - `age_deploy.prototxt`  
   - `age_net.caffemodel`  
   - `gender_deploy.prototxt`  
   - `gender_net.caffemodel`  
   - `Emotion_Detection.h5`  
   - `haarcascade_frontalface_default.xml` (for emotion detection face detection)

4. **Launch It**  
   ```bash
   python combined_detection_gradio.py
   ```
   *For emotion detection only, run `python test.py` instead.*

5. **Open the Fun**  
   Head to `http://127.0.0.1:7860` in your browser for the Gradio UI, or watch the webcam feed directly if using `test.py`.

---

## How to Use It

### Live Webcam Magic
1. Jump to the "Live Webcam" tab in the Gradio UI.
2. Hit **Start Webcam** to kick off the live stream.
3. Watch as faces are detected and labeled with age, gender, and emotion predictions in real-time.
4. Tap **Stop Webcam** to freeze the action—the last frame and results stay put.

**Terminal Sneak Peek**:
```
Gender: Female, confidence = 0.973
Age: (18-24), confidence = 0.821
Emotion: Happy, confidence = 0.905
Time: 0.298s
----------------------------------------
```

### Upload and Detect
1. Switch to the "Upload Image" tab.
2. Upload a face-containing image.
3. Click **Detect** and boom—results appear!
4. Check out the annotated image and detailed breakdown in the UI.

**UI Results**:
- Labeled image with predictions overlaid.
- Textbox spilling the beans on confidence scores and processing time.

---

## Contributing
Love it? Want to make it better? Here’s how:  
1. Fork this repo.  
2. Branch out: `git checkout -b my-cool-feature`.  
3. Commit your brilliance: `git commit -m 'Added something awesome'`.  
4. Push it: `git push origin my-cool-feature`.  
5. Open a pull request and let’s chat!

---

## License
This project is under the MIT License. Check out the [LICENSE](LICENSE) file for the nitty-gritty. Note: The emotion detection component is the intellectual property of Jerin Romijah Tuli. Unauthorized copying, modification, or distribution of that specific part is strictly prohibited. All rights reserved © Jerin Romijah Tuli.

---

## About the Developer
👩‍💻 **Jerin Romijah Tuli**  
📚 Rajshahi University of Engineering and Technology (RUET)  
🎓 Department of Computer Science and Engineering (CSE), 3rd Year  
📧 Email: your-ramijahtuli786@gmail.com  

**Areas of Interest**:  
- Data Science & AI  
- Machine Learning (ML)  
- App Development  
- Web Development (Django, Laravel)  

Aspiring to become a Data Scientist & AI Researcher, passionate about creating impactful tech solutions.

---

**Pro Tip**: Missing a model file? Double-check the [Get Started](#get-started) section. If something’s funky, peek at the terminal for clues. Have fun detecting!
```

This README combines both projects seamlessly, maintaining all original content while enhancing readability with Markdown formatting. You can copy and paste this directly into your `README.md` file. Let me know if you’d like any additional tweaks!
