<h1 align="center"><b>SafeDrive</b></h1>
![GIF](/media/demo.gif)
<hr>
<ul><h3><b>About the project</b></h3></ul>
This Python app was developed as a Capstone Project for Strive School AI Engineering bootcamp.
<br><br>

The idea of the project was to implement a lightweight solution to check driver's drowsiness, head orientation and detect cellphone usage.
<hr>
<ul><h3><b>How it works</b></h3></ul>
The script uses <a href="https://google.github.io/mediapipe/">MediaPipe</a> library to detect facial landmarks and <a href="https://www.tensorflow.org/">Tensorflow</a> for object detection.
<hr>
<ul><h4><b>MediaPipe implementation</b></h4></ul>
MediaPipe is used to extract facial landmarks that allow us to detect eyes and other keypoints. An algorith based on Eye Aspect Ratio (check <a href="http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf">this paper</a> for more information about it) is used to detect drowsiness. Head pose estimation is done by mathematical calculation on camera parameters and relevant keypoints detected on the face.
<br><br>
<ul><h4><b>Tensorflow Object Detection with MobileNet V3</b></h4></ul>
The script uses a MobileNet V3 architecture pretrained on MS Coco Dataset and filtered to obtain only bounding boxes for mobile phones. A MobileNet V2 fine tuning has been done on a sample from OpenImage v6 pictures but the performance of MobileNet V3 are better, so I decided to use that for the project. The fine tuning needs more data and computational power to obtain better result: feel free to contact me for more informaitons or for the checkpoints if you want to further train it.
<hr>
<ul><h3><b>Usage</b></h3></ul>
<ul>
<li>Clone this repository and install the required packages through pip.</li>

```
pip install -r requirements.txt
```

<li>Launch the script from the command line, giving the index of the webcam you want to use as -i argument. Paths to configuration file, weights and cateogries names are configured by default, but you can specify it as follow:

```
python safedrive.py -c path/to/config.pbtxt -w path/to/weights.pb -n path/to/names -i 0
```

 Camera index changes from system to system and depends also on the number of webcam you have, usually 0 is for the main one but if it doesn't work you may have to test it with 1, 2, and so on.</li>
 You can also download the entire mobilenet_v3 model from <a href="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz">here</a> and extract it wherever you want, but keep in mind that you'll need the file <i>coco.names</i> present in the "models" folder.
 
<br><br>
If you want to test single features, in the scripts folder there are splitted python files for drowsiness detection and head pose estimation.