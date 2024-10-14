Note that:

- only the composite_images are specified in test.jsonl
- source (composite) images are named foo_AA_BB.png
- masks are foo_AA.png 
- real images (which must be present but content is only used for
  evaluation, so for generation can be anything, I have used a copy
  of the masks) are named foo.jpg
- where 'foo' is a string, and 'AA' and 'BB' are integers
- real images must be JPEG, the source and mask must be PNG
    
For example:

```
data/Tristan/composite_images/goya1024_1_1.png
data/Tristan/composite_images/goya512_1_1.png
data/Tristan/masks/goya1024_1.png
data/Tristan/masks/goya512_1.png
data/Tristan/real_images/goya1024.jpg
data/Tristan/real_images/goya512.jpg
```

