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

def extract_digits_from_roi(test_img_roi):
    """提取测试ROI中的数字候选区域"""
    if len(test_img_roi.shape) == 3:
        gray = cv2.cvtColor(test_img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = test_img_roi.copy()

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    bin_roi = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 9)

    contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / h if h != 0 else 0
        if 10 < w < test_img_roi.shape[1]//2 and 10 < h < test_img_roi.shape[0]//2 \
           and 100 < area < 1000 and 0.2 < aspect_ratio < 1.0:
            digit_candidates.append((x, y, w, h))

    digit_candidates = sorted(digit_candidates, key=lambda r: r[0])

    print(f"检测到数字候选区域数量: {len(digit_candidates)}")
    return digit_candidates

def match_candidate_with_templates(candidate_img, digit_templates):
    """
    输入：候选数字图像，模板数字列表
    输出：最佳匹配模板索引和匹配分数
    """
    candidate_bin = preprocess_image(candidate_img, invert=True)
    best_score = -1
    best_idx = -1
    for idx, (digit_img, _) in enumerate(digit_templates):
        digit_bin = preprocess_image(digit_img, invert=True)
        # 缩放模板到候选大小，保证尺寸匹配
        digit_bin_resized = cv2.resize(digit_bin, (candidate_bin.shape[1], candidate_bin.shape[0]))
        res = cv2.matchTemplate(candidate_bin, digit_bin_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_idx = idx
    return best_idx, best_score

def recognize_digits_in_roi(test_img, digit_templates, top_left, digit_roi, draw_img):
    """
    识别测试图中某个数字ROI对应区域的数字，并在draw_img上绘制结果
    test_img: 测试彩色图
    digit_templates: 模板数字列表
    top_left: 模板整体匹配到测试图的左上角坐标 (x0, y0)
    digit_roi: 模板中某个数字的ROI (x, y, w, h)
    draw_img: 在此图上绘制识别结果（一般为测试图副本）
    """
    x0, y0 = top_left
    x, y, w, h = digit_roi
    # 计算测试图对应数字区域
    test_roi = test_img[y0 + y : y0 + y + h, x0 + x : x0 + x + w]

    # 提取测试ROI内的数字候选区域
    digit_candidates = extract_digits_from_roi(test_roi)

    for i, (cx, cy, cw, ch) in enumerate(digit_candidates):
        candidate_img = test_roi[cy:cy+ch, cx:cx+cw]
        idx, score = match_candidate_with_templates(candidate_img, digit_templates)
        print(f"候选数字{i}匹配模板数字{idx}，得分{score:.3f}")

        # 在draw_img上绘制候选数字框和匹配数字索引（坐标需转换到draw_img坐标系）
        abs_x1 = x0 + x + cx
        abs_y1 = y0 + y + cy
        abs_x2 = abs_x1 + cw
        abs_y2 = abs_y1 + ch

        cv2.rectangle(draw_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
        cv2.putText(draw_img, str(idx), (abs_x1, abs_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    template_bin = preprocess_image(template_img, invert=True)
    test_bin = preprocess_image(test_img, invert=True)

    cv2.imshow("Template Binary", template_bin)
    cv2.imshow("Test Image Binary", test_bin)

    # 再次确认尺寸
    if template_bin.shape[0] > test_bin.shape[0] or template_bin.shape[1] > test_bin.shape[1]:
        print("警告：模板尺寸大于测试图像尺寸，无法匹配")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 整体模板匹配
    result = cv2.matchTemplate(test_bin, template_bin, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"最大匹配分数: {max_val:.3f}，位置: {max_loc}")

    h, w = template_bin.shape
    test_img_draw = test_img.copy()
    cv2.rectangle(test_img_draw, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
    cv2.imshow("Match Result", test_img_draw)

    # 提取数字模板
    digit_templates = extract_digit_templates(template_img)
    print(f"提取到数字模板数量: {len(digit_templates)}")

    # 在测试图副本上绘制所有数字识别结果
    for idx, (_, digit_roi) in enumerate(digit_templates):
        print(f"\n识别模板中第{idx}个数字对应测试图区域：")
        cv2.imshow(f"Template Digit {idx}", digit_templates[idx][0])
        recognize_digits_in_roi(test_img, digit_templates, max_loc, digit_roi, test_img_draw)

    # 一次性显示所有数字识别结果
    cv2.imshow("All Digits Recognition", test_img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 修改为你的模板和测试图像路径
    template_path = r"2.png"
    test_img_path = r"1.png"

    test_template_match(template_path, test_img_path)
