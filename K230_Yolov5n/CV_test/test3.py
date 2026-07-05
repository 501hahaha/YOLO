import cv2
import numpy as np

def preprocess_image(img, invert=True):
    """统一预处理：灰度+自适应阈值+反色（可选）"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 9)
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary


def extract_digit_templates(template_img):
    """提取模板图像中的数字区域，返回[(数字图像, (x,y,w,h)), ...]"""
    bin_img = preprocess_image(template_img, invert=True)
    cv2.imshow("Template Binary", bin_img)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_img_inv = cv2.bitwise_not(bin_img)#反向二值化获取黑字白底图片
    digit_templates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        digit_roi = bin_img_inv[y:y+h, x:x+w]
        digit_templates.append((digit_roi, (x, y, w, h)))

    digit_templates = sorted(digit_templates, key=lambda x: x[1][0])  # 按x排序
    return digit_templates

def extract_regions_from_thresh(thresh_img):
    # thresh_img 应该是二值图，白色数字黑色背景或反之，根据需要反转
    # 这里假设数字是白色，背景是黑色，如果不是，可以反转：
    thresh_img = cv2.bitwise_not(thresh_img)
    thresh_inv = cv2.bitwise_not(thresh_img)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        roi = thresh_inv[y:y+h, x:x+w]
        regions.append((roi, (x, y, w, h)))

    regions = sorted(regions, key=lambda x: x[1][0])  # 按x坐标排序
    return regions

def resize_to_fixed(img, size=(28, 28)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compare_digits(test_digits, template_digits):
    results = []
    for i, (test_img, test_rect) in enumerate(test_digits):
        test_resized = resize_to_fixed(test_img)
        best_score = -1
        best_idx = -1
        for j, (tmpl_img, tmpl_rect) in enumerate(template_digits):
            tmpl_resized = resize_to_fixed(tmpl_img)
            # 使用模板匹配，计算相似度
            res = cv2.matchTemplate(test_resized, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_idx = j
        results.append((i, best_idx, best_score))
    return results




def test_template_match(template_path, test_img_path):
    template_img = cv2.imread(template_path)
    test_img = cv2.imread(test_img_path)

    if template_img is None:
        print(f"无法读取模板图片: {template_path}")
        return
    if test_img is None:
        print(f"无法读取测试图片: {test_img_path}")
        return

    print(f"原始模板尺寸: {template_img.shape}")
    print(f"测试图像尺寸: {test_img.shape}")

    # 如果模板比测试图像大，缩放模板
    th, tw = template_img.shape[:2]
    ih, iw = test_img.shape[:2]
    if th > ih or tw > iw:
        scale_h = ih / th
        scale_w = iw / tw
        scale = min(scale_h, scale_w, 1.0)  # 缩放比例不超过1
        new_size = (int(tw * scale), int(th * scale))
        print(f"缩放模板尺寸为: {new_size}")
        template_img = cv2.resize(template_img, new_size)

    
    digit_templates = extract_digit_templates(template_img)  # 传入原始模板图像
    for i, (digit_img, (x, y, w, h)) in enumerate(digit_templates):
        cv2.imshow(f"Digit {i}", digit_img)
    template_bin = preprocess_image(template_img, invert=True)
    test_bin = preprocess_image(test_img, invert=True)
        
    # cv2.imshow("Template Binary", template_bin)
    cv2.imshow("Test Image Binary", test_bin)
    


    # 再次确认尺寸
    if template_bin.shape[0] > test_bin.shape[0] or template_bin.shape[1] > test_bin.shape[1]:
        print("警告：模板尺寸大于测试图像尺寸，无法匹配")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    result = cv2.matchTemplate(test_bin, template_bin, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"最大匹配分数: {max_val:.3f}，位置: {max_loc}")

    h, w = template_bin.shape
    test_img_draw = test_img.copy()
    # cv2.rectangle(test_img_draw, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
    # cv2.imshow("Match Result", test_img_draw)
    
    matched_region = test_img_draw[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
    cv2.imshow("Matched Region", matched_region)
    gray = cv2.cvtColor(matched_region, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, -9)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow("Thresholded Region", thresh)
    
    digit_thresh = extract_regions_from_thresh(thresh)  # 传入原始模板图像
    for i, (digit_threshimg, (x, y, w, h)) in enumerate(digit_thresh):
        cv2.imshow(f"Digit {i}", digit_threshimg)
        
   # 调用示例
    results = compare_digits(digit_thresh, digit_templates)

    filtered_results = [(test_idx, tmpl_idx, score) for test_idx, tmpl_idx, score in results if score > 0.7]

    # 提取所有tmpl_idx，去重（如果需要）
    tmpl_indices = [tmpl_idx for _, tmpl_idx, _ in filtered_results]
    # 如果想去重，可以用：
    # tmpl_indices = list(set(tmpl_indices))

    # 拼接成一行字符串
    tmpl_indices_str = " ".join(str(idx) for idx in tmpl_indices)

    print(f"匹配的模板索引有：{tmpl_indices_str}")

    # 显示图像部分可选
    # for test_idx, tmpl_idx, score in filtered_results:
    #     cv2.imshow(f"Test Digit {test_idx}", resize_to_fixed(digit_thresh[test_idx][0]))
    #     cv2.imshow(f"Template Digit {tmpl_idx}", resize_to_fixed(digit_templates[tmpl_idx][0]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # 修改为你的模板和测试图像路径
    template_path = r"D:\Users\a3828\Desktop\K230_Yolov5n\CV_test\2.png"
    test_img_path = r"D:\Users\a3828\Desktop\K230_Yolov5n\CV_test\1.png"

    test_template_match(template_path, test_img_path)