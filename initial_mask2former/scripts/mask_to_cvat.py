import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from xml.dom import minidom

def get_args():
    parser = argparse.ArgumentParser(description="Convert segment masks to CVAT XML.")
    # nargs='+' allows accepting multiple arguments separated by spaces
    parser.add_argument("--input-dir", type=str, required=True, nargs='+',
                        help="The base directory containing dataloader_output and segmentation_output")
    return parser.parse_args()


# Your specific palette (Background [0,0,0] is omitted because we don't label it in CVAT)
COLORS = {
    "cracks": [250, 50, 83],
    "cracks_alligator": [36, 179, 83],
    "cracks_severe": [102, 255, 102],
    "edge_breaks": [255, 0, 255],
    "fretting": [204, 153, 51],
    "pothole": [115, 51, 128],
    "manhole": [34, 62, 209],
    "pole_shadow": [172, 84, 109],
}

def create_cvat_xml(base_dir):
    
    image_dir = os.path.join(base_dir, "dataloader_output/images/")
    mask_dir = os.path.join(base_dir, "segmentation_output/defectmasks")
    output_xml = os.path.join(base_dir, "cvat_annotations.xml")
    
    # Initialize the CVAT XML Structure
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    
    # Add your classes to the XML metadata
    for cls_name in COLORS.keys():
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = cls_name

    # Get list of images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])

    for img_id, img_filename in enumerate(image_files):
        print(f"Processing {img_filename}...")
        
        # Load original image to get dimensions
        img_path = os.path.join(image_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape
        
        # Create image tag in XML
        xml_image = ET.SubElement(annotations, "image", id=str(img_id), name=img_filename, width=str(width), height=str(height))
        
        # Load corresponding mask
        # Assumes mask is named exactly the same but ends in .png
        mask_filename = os.path.splitext(img_filename)[0] + "_blend.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"  -> No mask found for {img_filename}, skipping.")
            continue
            
        # Read mask and convert to RGB
        mask = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask_rgb = cv2.resize(mask_rgb, (width, height), interpolation=cv2.INTER_NEAREST)

        # Find polygons for each class
        for cls_name, color in COLORS.items():
            # Create a binary mask where pixels match the target color
            lower = np.array(color)
            upper = np.array(color)
            binary_mask = cv2.inRange(mask_rgb, lower, upper)
            
            # Find contours (outlines) of the shapes
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # Ignore tiny specks of noise (less than 10 pixels area)
                if cv2.contourArea(cnt) < 10: 
                    continue
                
                # SIMPLIFY THE POLYGON (Crucial for cracks so CVAT doesn't freeze)
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Convert points to CVAT string format: "x,y;x,y;x,y"
                points_str = ";".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
                
                # Write to XML
                ET.SubElement(xml_image, "polygon", label=cls_name, points=points_str)

    # Save formatted XML to disk
    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    with open(output_xml, "w") as f:
        f.write(xml_str)
    print(f"\nSuccess! Saved {output_xml}")
    
if __name__ == "__main__":
    args = get_args()
    
    raw_inputs = args.input_dir
    clean_folders = []
    for item in raw_inputs:
        cleaned = item.replace('[', '').replace(']', '').replace(',', '').strip()
        if cleaned: # only add if it's not an empty string
            clean_folders.append(cleaned)
    
    for folder in clean_folders:
        # Construct the path inside your data folder
        base_path = "data/laura_tfm_sun_22_dry" 
        target_dir = os.path.join(base_path, folder)
        
        print(f"\n--- Starting processing for: {target_dir} ---")
        
        if os.path.exists(target_dir):
            create_cvat_xml(target_dir)
        else:
            print(f"Error: Directory '{target_dir}' not found. Skipping.")