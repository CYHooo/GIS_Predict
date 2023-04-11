# Error's [INFO]: error: (-215:Assertion failed) !dsize.empty() in function 'resize'
change
> src/utils/augmentation.py

in line 325:

    masks = cv2.resize(masks, (width, height))

to:

    cv_limit = 512
    if masks.shape[2] <= cv_limit:
        masks = cv2.resize(masks, (width, height))
    else:
        # split masks array on batches with max size 512 along channel axis, resize and merge them back
        masks = np.concatenate([cv2.resize(masks[:, :, i:min(i + cv_limit, masks.shape[2])], (width, height))
                                for i in range(0, masks.shape[2], cv_limit)], axis=2)
---

# Error's [INFO]: shape not matching(i.e., shape(100,4) can not into shape(100,6))
change
> src/utils/dataloader.py

in line 249:

    boxes = boxes[:,:-2]
