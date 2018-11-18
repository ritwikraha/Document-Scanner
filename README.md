# Document-Scanner
A document scanner built using Open-CV python
The application uses morphological gradient and adaptive gaussian thresholding to succesfully detect edges and find contours.
The contours are then used to warp the document in the picture to a suitable aspect ratio to which different thresholding techniques are applied.
