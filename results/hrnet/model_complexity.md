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