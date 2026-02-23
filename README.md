# Detecting-cylinder-microrobots-in-ultrasound-images
Introducing our deep learning system for detecting cylinder microrobots in ultrasound images — trained on USMicroMagSet, a first-of-its-kind dataset of 40,000 ultrasound frames capturing 8 different microrobot types navigating microfluidic channels.

![](microbot.gif)





🔬 We just built something that sits at the intersection of AI, robotics, and medicine — and it's wild.

📡 Introducing our deep learning system for detecting cylinder microrobots in ultrasound images — trained on USMicroMagSet, a first-of-its-kind dataset of 40,000 ultrasound frames capturing 8 different microrobot types navigating microfluidic channels.

🤖 Why does this matter?

Microrobots are the future of minimally invasive medicine — imagine tiny robots navigating your bloodstream to deliver drugs, clear blockages, or perform targeted biopsies. But before they reach the clinic, we need AI systems that can SEE them in real time inside the human body.

That's exactly what we built.

⚙️ What's under the hood:

→ YOLOv5 trained specifically on SMF-class Cylinder microrobots

→ 93.2% mAP@0.5 | Precision: 0.91 | Recall: 0.88 | F1: 0.89

→ <8ms inference per frame (real-time capable)

→ Siemens 14L5 SP ultrasound probe at 14 MHz

→ Full Flask API backend + interactive web dashboard

→ Bird's-eye trajectory view, live detection feed & analytics

🏥 For Doctors & Radiologists:

This system could one day plug directly into your ultrasound workflow — flagging microrobot position automatically, so you can focus on the procedure, not the screen.

👨‍💻 For Engineers & Researchers:

The full stack is open — YOLOv5 weights, Flask inference API, drag-and-drop image analyzer, precision-recall curves, training analytics. Reproducible. Extensible. Ready to benchmark.

📊 Dataset highlights:

• 3 locomotion classes: SMF / RMF / OMF

• 8 robot types: Sphere, Cube, Cylinder, Helical, Soft Sheet, Rolling Cube, Flagella, Chainlike

• 40K images extracted from 10-min videos per robot

• Recorded at NSTP Islamabad with clinical-grade US equipment

🚀 This is just the beginning. Next: multi-class detection, 3D tracking, and integration with magnetic field control systems.

If you work in medical robotics, surgical AI, ultrasound imaging, or autonomous microrobotics — let's connect. 🤝

#MedicalRobotics #AIinHealthcare #ComputerVision #YOLOv5 #Microrobotics #DeepLearning #UltrasoundImaging #MedTech #BiomedicalEngineering #RoboticsResearch #SurgicalAI #MachineLearning #MedicalImaging #HealthTech #NSTP #Mechatronics #SmartHealthcare #RealTimeAI #MinimallyInvasive #FutureOfMedicine
