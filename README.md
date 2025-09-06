🚦 Real-Time Vehicle Detection, Classification, and Counting

This project is a real-time computer vision system that detects, classifies, and counts vehicles crossing an intersection. It was inspired by my Civil Engineering coursework, where traffic volume studies were once done manually. With YOLOv8, BoT-SORT tracking, and OpenCV, this project automates that process and demonstrates how AI can support traffic monitoring, smart cities, and civil engineering applications.

✨ Features

Detects and classifies cars, motorcycles, buses, and trucks

Tracks vehicles across frames using BoT-SORT to avoid double-counting

Counts vehicles as they cross a predefined line or region of interest (ROI)

Real-time visualization with:

Bounding boxes

Class labels + confidence scores

Tracking IDs

Live counters for each class

Configurable input (video file or webcam feed)

FPS monitoring for performance insights

🛠️ Tech Stack

Python

Ultralytics YOLOv8
 – Object Detection

BoT-SORT
 – Multi-Object Tracking

OpenCV – Video processing and visualization

NumPy – Numerical operations

📂 Project Structure
count_traffic.py   # Main script for detection, tracking, and counting
intersection.mp4   # Sample input video (replace with your own)

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/yourusername/vehicle-counter.git
cd vehicle-counter

2. Install Dependencies

It’s recommended to use a virtual environment.

pip install ultralytics opencv-python numpy lap filterpy

3. Run the Program
python count_traffic.py


By default, it uses intersection.mp4 as input. To use a webcam instead:

VIDEO_SOURCE = 0

4. Customize

Adjust the counting line in LINE_PTS

Define a polygonal ROI if needed (ROI_POLY)

Change the list of allowed vehicle classes (ALLOWED_LABELS)

📊 Example Output

The program overlays bounding boxes, class labels, and live counts on the video feed:

car: 10

motorcycle: 5

bus: 2

truck: 3

🎓 Background Story

In my Civil Engineering studies, I once had to manually count vehicles at intersections to determine Level of Service (LOS) for redesign projects. That meant hours of standing on the roadside with a tally sheet. Today, with AI, this tedious task can be automated. This project showcases how computer vision can bridge civil engineering and AI engineering.

⚠️ Limitations

Occasional misclassifications (e.g., cars vs SUVs)

Missed counts in heavy occlusion scenarios

Performance depends on video quality and angle

🔮 Future Improvements

Fine-tune YOLOv8 on a local dataset of Philippine vehicles (e.g., jeepneys, tricycles)

Directional counting (northbound vs southbound)

Integration with dashboards (e.g., Streamlit, Power BI)

Cloud or edge deployment for real-world intersections

📜 License

This project is licensed under the MIT License.
