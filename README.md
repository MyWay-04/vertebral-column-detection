This project focuses on spine segmentation from X-ray images using classical image processing techniques.
The pipeline begins with preprocessing, where masking is applied to remove irrelevant regions and gamma correction is used to enhance contrast. The image is then smoothed using average filtering and sharpened to highlight important structures.
Next, a segmentation-based thresholding method is applied to extract potential spine regions. Multiple masks are combined using bitwise operations to improve robustness.
After that, connected component analysis is used to identify candidate regions. These regions are further refined using region growing based on intensity similarity, followed by split-and-merge techniques to improve structural consistency.
Finally, each segmented component is visualized using a distinct colormap and overlaid on the original X-ray image.
This approach demonstrates how traditional techniques such as thresholding, connected components, and region-based segmentation can effectively be used for medical image analysis.

**▶️How to RUN**

pip install -r requirements.txt

python main.py

## Techniques Used
- Gamma correction
- Thresholding
- Connected component analysis
- Region growing
- Split-and-merge segmentation
