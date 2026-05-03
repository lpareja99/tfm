
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