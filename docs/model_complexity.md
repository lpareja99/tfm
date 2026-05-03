Model Compxity and speed results:

- Swin:

{
    "config": "config.py",
    "unit": "img / s",
    "overall_fps_1": 15.35,
    "average_fps": 15.35,
    "fps_variance": 0.0
}

==============================
COMPLEXITY RESULTS
==============================
Full Model Parameters: 47.28 M
Backbone GFLOPs: 25.61 G
==============================

Detailed Parameter Breakdown:
| name                            | #elements or shape   |
|:--------------------------------|:---------------------|
| model                           | 47.3M                |
| backbone                        | 27.5M                |
| backbone.patch_embed            | 4.9K                 |
| backbone.stages                 | 27.5M                |
| backbone.norm0                  | 0.2K                 |
| backbone.norm1                  | 0.4K                 |
| backbone.norm2                  | 0.8K                 |
| backbone.norm3                  | 1.5K                 |
| decode_head                     | 19.8M                |
| decode_head.pixel_decoder       | 5.4M                 |
| decode_head.transformer_decoder | 14.2M                |
| decode_head.query_embed         | 25.6K                |
| decode_head.query_feat          | 25.6K                |
| decode_head.level_embed         | 0.8K                 |
| decode_head.cls_embed           | 2.6K                 |
| decode_head.mask_embed          | 0.1M                 |

---

- HRNet:

{
    "config": "config.py",
    "unit": "img / s",
    "overall_fps_1": 6.42,
    "average_fps": 6.42,
    "fps_variance": 0.0
}

==============================
COMPLEXITY RESULTS
==============================
Full Model Parameters: 48.94 M
Backbone GFLOPs: 41.53 G
==============================

Detailed Parameter Breakdown:
| name                            | #elements or shape   |
|:--------------------------------|:---------------------|
| model                           | 48.9M                |
| backbone                        | 29.3M                |
| backbone.conv1                  | 1.7K                 |
| backbone.bn1                    | 0.1K                 |
| backbone.conv2                  | 36.9K                |
| backbone.bn2                    | 0.1K                 |
| backbone.layer1                 | 0.3M                 |
| backbone.transition1            | 0.2M                 |
| backbone.stage2                 | 0.4M                 |
| backbone.transition2            | 74.0K                |
| backbone.stage3                 | 6.8M                 |
| backbone.transition3            | 0.3M                 |
| backbone.stage4                 | 21.2M                |
| decode_head                     | 19.6M                |
| decode_head.pixel_decoder       | 5.2M                 |
| decode_head.transformer_decoder | 14.2M                |
| decode_head.query_embed         | 25.6K                |
| decode_head.query_feat          | 25.6K                |
| decode_head.level_embed         | 0.8K                 |
| decode_head.cls_embed           | 2.6K                 |
| decode_head.mask_embed          | 0.2M                 |

---

- InterImage:

{
    "config": "config.py",
    "unit": "img / s",
    "overall_fps_1": 7.16,
    "average_fps": 7.16,
    "fps_variance": 0.0
}

==============================
COMPLEXITY RESULTS
==============================
Full Model Parameters: 48.53 M
Backbone GFLOPs: 25.10 G
==============================

Detailed Parameter Breakdown:
| name                            | #elements or shape   |
|:--------------------------------|:---------------------|
| model                           | 48.5M                |
| backbone                        | 28.8M                |
| backbone.patch_embed            | 19.6K                |
| backbone.levels                 | 28.7M                |
| decode_head                     | 19.8M                |
| decode_head.pixel_decoder       | 5.3M                 |
| decode_head.transformer_decoder | 14.2M                |
| decode_head.query_embed         | 25.6K                |
| decode_head.query_feat          | 25.6K                |
| decode_head.level_embed         | 0.8K                 |
| decode_head.cls_embed           | 2.6K                 |
| decode_head.mask_embed          | 0.2M                 |

---

- Flash Intern Image:

{
    "config": "config_tiny.py",
    "unit": "img / s",
    "overall_fps_1": 14.64,
    "average_fps": 14.64,
    "fps_variance": 0.0,
    "total_parameters_M": 50.24
}

==============================
COMPLEXITY RESULTS
==============================
Full Model Parameters: 50.54 M
Backbone GFLOPs: 26.53 G
==============================

Detailed Parameter Breakdown:
| name                            | #elements or shape   |
|:--------------------------------|:---------------------|
| model                           | 50.5M                |
| backbone                        | 30.8M                |
| backbone.patch_embed            | 19.6K                |
| backbone.levels                 | 30.8M                |
| decode_head                     | 19.8M                |
| decode_head.pixel_decoder       | 5.3M                 |
| decode_head.transformer_decoder | 14.2M                |
| decode_head.query_embed         | 25.6K                |
| decode_head.query_feat          | 25.6K                |
| decode_head.level_embed         | 0.8K                 |
| decode_head.cls_embed           | 2.6K                 |
| decode_head.mask_embed          | 0.2M                 |

---

- BeiT v2:

{
    "config": "config.py",
    "unit": "img / s",
    "overall_fps_1": 10.64,
    "average_fps": 10.64,
    "fps_variance": 0.0
}

==============================
COMPLEXITY RESULTS
==============================
Full Model Parameters: 109.02 M
Backbone GFLOPs: 107.12 G
==============================

Detailed Parameter Breakdown:
| name                            | #elements or shape   |
|:--------------------------------|:---------------------|
| model                           | 0.1G                 |
| backbone                        | 86.2M                |
| backbone.cls_token              | (1, 1, 768)          |
| backbone.patch_embed            | 0.6M                 |
| backbone.layers                 | 85.6M                |
| neck                            | 3.1M                 |
| neck.lateral_convs              | 0.8M                 |
| neck.convs                      | 2.4M                 |
| decode_head                     | 19.6M                |
| decode_head.pixel_decoder       | 5.3M                 |
| decode_head.transformer_decoder | 14.2M                |
| decode_head.query_embed         | 25.6K                |
| decode_head.query_feat          | 25.6K                |
| decode_head.level_embed         | 0.8K                 |
| decode_head.cls_embed           | 2.6K                 |
| decode_head.mask_embed          | 0.1M                 |