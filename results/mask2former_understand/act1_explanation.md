### The Big Picture of Act 1
In this act, the model is acting like a team of cartographers mapping an unknown territory . It is taking your raw, high-resolution road image and systematically destroying the "pixels" to extract the mathematical "meaning." 

The goal of Act 1 is to create **Panel 6 (The Master Map)**, which is the perfect, 256-dimensional search space that the 100 queries will use in Act 2. 

Here is how the data flows from Panel 1 to Panel 6.

---

### Phase A: The Backbone (Panels 1, 2, and 3)
The Swin-T backbone's job is to look at the image at multiple scales. As you move from Layer 1 to Layer 4, the resolution drops dramatically, but the "understanding" deepens.

* **Panel 1: Backbone Layer 1 (1/4 Scale)**
    * **What it is:** The highest-resolution feature map. 
    * **What is happening:** The model is looking at tiny $4 \times 4$ pixel windows. It has no idea what a "pothole" is yet. It is only looking for sharp color gradients, harsh lines, and high-frequency textures (like the grit of the asphalt). If there is a crack, this layer will light up exactly on the edges of that crack.
* **Panel 2: Backbone Layer 3 (1/16 Scale)**
    * **What it is:** A medium-resolution feature map.
    * **What is happening:** The model has zoomed out. It is combining the lines and edges from Panel 1 into regional patterns. It stops caring about individual pebbles and starts identifying "zones" of similar texture, like the rough zone of a fretting patch versus smooth pavement.
* **Panel 3: Backbone Layer 4 (1/32 Scale)**
    * **What it is:** The lowest-resolution, most abstract feature map.
    * **What is happening:** The model has zoomed out as far as it can. It is practically blind to crisp edges, but it understands **global context**. It knows the spatial relationship between things (e.g., "There is a large dark anomaly in the bottom left, and it is surrounded by uniform gray"). 

### Phase B: The Missing Link (Panel 5)
Now, the **Pixel Decoder** takes over. It cannot just stack these layers together, because Panel 1 is huge and Panel 3 is tiny. 

* **Panel 5: The Transformer Encoder (The Missing Link)**
    * **What it is:** This is the lowest-resolution layer (Panel 3) *after* it has been passed through a Transformer Encoder (Multi-Scale Deformable Attention) . 
    * **What is happening:** The deep, blurry context layer is allowed to "look" at the sharper layers. The model is essentially saying, *"I know there is a general anomaly in the bottom left (Panel 3), let me double-check the textures in Panel 2 to confirm it's not just a shadow."* Panel 5 represents a highly intelligent, context-aware, but still very low-resolution map.

### Phase C: The Merger (Panel 6)
* **Panel 6: The Final Master Map**
    * **What it is:** The 256-dimensional Fused Feature Space. 
    * **What is happening:** The Pixel Decoder takes that super-smart, low-resolution map from Panel 5 and progressively scales it back up (upsampling). As it scales up, it adds the crisp, high-resolution lines and edges from Panel 1 back into it. 
    * **The Correlation:** Panel 6 is the perfect marriage of Panel 1 and Panel 5. It has the **crisp, exact physical boundaries** of the edges (Panel 1), combined with the **deep contextual understanding** of what those shapes actually mean (Panel 5).

By the end of Act 1, the raw RGB colors (Panel 4) have been entirely discarded. Every pixel on your screen is now represented by this Master Map—a mathematical recipe that the 100 detectives are about to interrogate.