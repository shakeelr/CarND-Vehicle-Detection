## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_visualization.png
[image2]: ./output_images/hog_examples.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/hog_predictions.png
[image5]: ./output_images/hog_hitboxes.png
[image6]: ./output_images/window_predictions.png
[image7]: ./output_images/nn_hitboxes.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/thresholded.png
[image10]: ./output_images/car_detect.png
[image11]: ./output_images/test_examples.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  Note that there are two IPython notebooks in this submission, as I started doing the project using HOG features and an SVM classifier in the notebook titled `P5_HOG.ipynb`, but wasn't satisfied with the performance and switched to a deep learning approach contained in the notebook titled `P5_NN.ipynb`. While the `P5_HOG.ipynb` notebook contains functioning code, only the`P5_NN.ipynb` contains a complete pipeline implementation.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cells 2 through 5 of the IPython notebook `P5_HOG.ipynb`.

I started by reading in and labeling all the `vehicle` and `non-vehicle` images.  Here is an example of 32 random images from the resulting dataset:

![alt text][image1]

I then visualized HOG features on a sample of images in different color spaces to get a feel for what the `skimage.hog()` output looks like, as shown below.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The data was split into training and test datasets.  I then tried various combinations of HOG parameters in different colorspaces as seen in cells 6 and 7 of the IPython notebook `P5_HOG.ipynb`, training an SVC on each set of resulting HOG features and chose the one with the best accuracy on the test dataset.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The best SVC was found in the HSV colorspace, with 12 orientations, 16 pixels per block, and 4 blocks per cell.  I trained the final SVC using these parameters in cell 8 of the IPython notebook `P5_HOG.ipynb` and achieved a test accuracy of 0.9718.

### Sliding Window Search with HOG Classifier

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I defined a function `get_windows(img, y, scale, vis`) in cell 10 of `P5_HOG.ipynb` to get a row of 64x64 pixel windows from a given image, and return the resulting window images and coordinates for their corresponding boxes.  The function accepts the y coordinate for the top of the row of windows to be returned, and a scale factor, to allow searching at various scales.  An example of the boxes drawn on a sample image is shown below:

![alt text][image3]

And the resulting labels of the windows using the SVM classifier is shown below:

![alt text][image4]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the `get_windows(img, y, scale, vis`) to search the image on a variety of different scales, using the SVM classifier derived from the section above and marked all the hitboxes as shown below:

![alt text][image5]

Unfortunately, the result was not as good as I had hoped.  There were a significant number of false positives detected.  While I'm sure I could have improved my SVM classifier by adding binned color features, I decided to instead apply the deep learning techniques we learned earlier in this course and train a CNN to detect cars instead, as I felt a CNN would likely be a more robust model for detecting vehicles.

### Deep learning approach using a CNN

#### 1. CNN architecture and training

Similar to before, I started by reading in and labeling all the `vehicle` and `non-vehicle` images, and splitting the dataset into a train and test dataset.  However this time I used a model architecture (code cell 7 of `P5_NN.ipynb`) consisting of of a convolution neural network with layers as visualized in the table below:

| Layer					|Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 64,64,1 Grayscale Image						|
| Convolution 5x5		| 32 filters, subsample=(2,2), valid padding	|
| RELU					| 												|
| Convolution 3x3		| 64 filters, subsample=(2,2), valid padding	|
| RELU					| 												|
| Flatten				| 												|
| Dense					| Outputs 100  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 50  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 10  									|
| Batch Normalization	| 												|
| RELU					| 												|
| Dense					| Outputs 1  									|
| Sigmoid				| 												|

I defined a preprocessing function (code cell 4 of `P5_NN.ipynb`), and a generator function (code cell 5), then trained and saved the network in code cells 6-8. Finally I defined a predict function in code cell 9 to use in conjuction with my sliding window search.

### Sliding Window Search with CNN

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search (code cell 11 of `P5_NN.ipynb`) was implemented in the same way as before, and the same scales were chosen to search.  One minor change was the addition of weights to the windows.  I found that certain scales generated more false positives, so I assigned them less weight.  The windows, scales, and weights are implemented in code cell 13 of `P5_NN.ipynb`

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the `get_windows(img, y, scale, vis`) function to search the image on a variety of different scales  (code cell 13 of `P5_NN.ipynb`), using the CNN classifier derived from the section above and marked all the hitboxes as shown below:

![alt text][image7]

The performance achieved is much better than what I got using the HOG feature classifier, so I continued to use the CNN in my final pipeline implemented in code cell 16 of `P5_NN.ipynb` and described in the section on video implementation below.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and thresholded the heatmap the reduce false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap for a test image before and after thresholding, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid:

Heatmap:

![alt text][image8]

Thresholded heatmap:

![alt text][image9]

Labeling using `scipy.ndimage.measurements.label()` and bounding box overlaid

![alt text][image10]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I first attempted the project using a SVM classifier trained on HOG gradients but shifted to a deep learning approach that worked much better.  That said there is still significant room for improvement.  The CNN still detects a number of false positives in the video, and struggles to properly classify the white car at times, sometimes seeing it as 2-3 seperate cars.  Further turning the weights and threshold might help it see the white car better, but at the expense of more false positives.  A better solution would be to try augmenting the training data set with pictures of white cars similar to the one in the video, along with pictures of objects in the video generating false positives such as the railing on the side.

