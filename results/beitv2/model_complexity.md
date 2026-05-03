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