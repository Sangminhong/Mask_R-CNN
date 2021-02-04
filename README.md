# Mask_R-CNN

### Overview (What is Mask RCNN)
* Deep residual network which can solve instance segmentation
  * Instance segmentation referes to seperating different objects from a image or a video. 
  
* Two steps of Mask RCNN
  1. Generates proposals regarding to regions where the object might occupy.
  1. Predicts the class, bounding box, and mask in pixel level of the object.
  
* Backbone structure
   * Feature Pyramid Netwrok style deep nerual network.
   * Consists of bottom-up path way, top-bottom pathway, and lateral connections. <br/><br/> 
    <img src="Images/Feature pyramid networkk architecture.PNG" width="300" height="150" />
   
   
 ### Step 1
 * RPN scans all FPN path and prposes regions where it may contain objects
 * Anchors are set of boxes sharing the same center 
 * RPN uses the anchors to predict and output the coordinates and thes size of the object.
 
### Step 2
* ROI align locates the relevent areas of fetaure map
* One branch generates masks for each object in pixel level

### Architecture of Mask RCNN
<img src="Images/architecture of mask RCNN.PNG" width="400" height="300" />
