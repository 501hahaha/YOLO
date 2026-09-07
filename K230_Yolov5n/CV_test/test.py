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
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_templates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        digit_roi = template_img[y:y+h, x:x+w]
        digit_templates.append((digit_roi, (x, y, w, h)))

    digit_templates = sorted(digit_templates, key=lambda x: x[1][0])  # 按x排序
    return digit_templates

def multi_template_match(digit_templates, test_img, threshold=0.7):
    test_bin = preprocess_image(test_img, invert=True)
    matches = []

    print(f"测试图像尺寸: {test_img.shape[1]}x{test_img.shape[0]}")

    for idx, (digit_img, bbox) in enumerate(digit_templates):
        digit_bin = preprocess_image(digit_img, invert=True)
        h, w = digit_bin.shape

        print(f"模板{idx}尺寸: {w}x{h}")

        if h > test_bin.shape[0] or w > test_bin.shape[1]:
            print(f"模板{idx}尺寸大于测试图像，跳过匹配")
            continue

        result = cv2.matchTemplate(test_bin, digit_bin, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(f"模板{idx}最大匹配分数: {max_val:.3f}")

        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):  # (x,y)
            matches.append({
                'template_idx': idx,
                'position': pt,
                'size': (w, h),
                'score': result[pt[1], pt[0]]
            })

    return matches

def draw_matches(test_img, matches, digit_templates):
    img_draw = test_img.copy()
    for m in matches:
        x, y = m['position']
        w, h = m['size']
        idx = m['template_idx']
        score = m['score']

        # 只用绿色矩形框绘制匹配区域（和单模板匹配时一致）
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_draw, f"Idx:{idx} Score:{score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_draw

def single_template_match(template_img, test_img):
    th, tw = template_img.shape[:2]
    ih, iw = test_img.shape[:2]
    if th > ih or tw > iw:
        scale_h = ih / th
        scale_w = iw / tw
        scale = min(scale_h, scale_w, 1.0)
        new_size = (int(tw * scale), int(th * scale))
        print(f"缩放模板尺寸为: {new_size}")
        template_img = cv2.resize(template_img, new_size)

    template_bin = preprocess_image(template_img, invert=True)
    test_bin = preprocess_image(test_img, invert=True)

    if template_bin.shape[0] > test_bin.shape[0] or template_bin.shape[1] > test_bin.shape[1]:
        print("警告：模板尺寸大于测试图像尺寸，无法匹配")
        return None

    result = cv2.matchTemplate(test_bin, template_bin, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"最大匹配分数: {max_val:.3f}，位置: {max_loc}")

    h, w = template_bin.shape
    test_img_draw = test_img.copy()
    cv2.rectangle(test_img_draw, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)

    return test_img_draw

if __name__ == "__main__":
    # 修改为你的模板和测试图像路径
    template_path = r"2.png"
    test_img_path = r"1.png"

    template_img = cv2.imread(template_path)
    test_img = cv2.imread(test_img_path)

    if template_img is None or test_img is None:
        print("无法读取图片")
        exit()

    use_multi_template = True  # True:多模板匹配，False:单模板匹配

    if use_multi_template:
        digit_templates = extract_digit_templates(template_img)
        print(f"提取到数字模板数量: {len(digit_templates)}")

        threshold = 0.7
        matches = multi_template_match(digit_templates, test_img, threshold=threshold)
        print(f"匹配结果数量: {len(matches)}")

        if len(matches) == 0:
            print("没有匹配到任何数字")

        img_result = draw_matches(test_img, matches, digit_templates)
        cv2.imshow("Multi Template Match Result", img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        img_result = single_template_match(template_img, test_img)
        if img_result is not None:
            cv2.imshow("Single Template Match Result", img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
