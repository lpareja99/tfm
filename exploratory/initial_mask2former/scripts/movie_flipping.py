from moviepy import VideoFileClip

# 1. Load your GoPro video
clip = VideoFileClip("GX010165.MP4")

# 2. Rotate it (v2.x uses 'rotated' instead of 'rotate')
# If the result is upside down later, change 90 to -90
fixed_clip = clip.rotated(90) 

# 3. Save the new file
# This will take a few minutes as it re-renders the pixels
fixed_clip.write_videofile("GX010165_landscape.mp4", codec="libx264")