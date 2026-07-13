import cv2
import numpy as np
from matplotlib import pyplot as plt


class SIFTROIDetector:
    def __init__(self, min_match_count=10, ratio_threshold=0.7):
        """
        Initialize SIFT ROI Detector

        Args:
            min_match_count: Minimum number of matches required
            ratio_threshold: Threshold for Lowe's ratio test
        """
        self.min_match_count = min_match_count
        self.ratio_threshold = ratio_threshold

        # Create SIFT detector
        self.sift = cv2.SIFT_create()
        # self.sift = cv2.ORB_create()
        # self.sift = cv2.xfeatures2d.SURF_create()

        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Store template information
        self.template = None
        self.template_roi = None
        self.template_kp = None
        self.template_des = None
        self.template_roi_corners = None

        # For debugging
        self.last_query_img = None
        self.last_query_kp = None
        self.last_query_des = None
        self.last_good_matches = None
        self.last_matches = None
        self.last_src_pts = None
        self.last_dst_pts = None

    def load_template(self, template_path, template_roi_path):
        """
        Load template image and ROI mask

        Args:
            template_path: Path to template image
            template_roi_path: Path to ROI mask (background 0, ROI 255)
        """
        # Read template image
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Cannot read template image: {template_path}")

        # Read ROI mask
        self.template_roi = cv2.imread(template_roi_path, cv2.IMREAD_GRAYSCALE)

        if self.template_roi is None:
            raise ValueError(f"Cannot read ROI mask: {template_roi_path}")

        # Binarize ROI mask
        # _, self.template_roi = cv2.threshold(self.template_roi, 127, 255, cv2.THRESH_BINARY)
        # self.template_roi = self.template_roi * 255

        # Extract ROI contour as reference
        contours, _ = cv2.findContours(self.template_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Get bounding box corners
            rect = cv2.boundingRect(largest_contour)
            x, y, w, h = rect
            self.template_roi_corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        self.template = self.template * self.template_roi
        cv2.imwrite('tmp.jpg', self.template)
        # Calculate SIFT features ONLY in ROI area
        self.template_kp, self.template_des = self.sift.detectAndCompute(
            self.template,
            mask=self.template_roi  # Only detect features in ROI area
            # mask=None  # Only detect features in ROI area
        )

        print(f"Template loaded, detected {len(self.template_kp)} features in ROI")

    def detect_roi(self, query_image_path):
        """
        Detect ROI in query image

        Args:
            query_image_path: Path to query image

        Returns:
            dict: Detection result dictionary
        """
        # Reset debug info
        self.last_query_img = None
        self.last_query_kp = None
        self.last_query_des = None
        self.last_good_matches = None
        self.last_matches = None
        self.last_src_pts = None
        self.last_dst_pts = None

        # Read query image
        query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            return {"success": False, "message": f"Cannot read query image: {query_image_path}"}

        # Save query image for visualization
        self.last_query_img = query_img

        # Calculate query image SIFT features
        self.last_query_kp, self.last_query_des = self.sift.detectAndCompute(query_img, None)
        # self.template_des = np.float32(self.template_des)
        # self.last_query_des = np.float32(self.last_query_des)

        if self.last_query_des is None or len(self.last_query_des) < 2:
            return {"success": False, "message": "Not enough features in query image"}

        # Feature matching
        if self.template_des is None or len(self.template_des) < 2:
            return {"success": False, "message": "Not enough features in template"}

        self.last_matches = self.flann.knnMatch(self.template_des, self.last_query_des, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in self.last_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        self.last_good_matches = good_matches
        print(f"Found {len(good_matches)} good matches")

        if len(good_matches) < self.min_match_count:
            return {
                "success": False,
                "message": f"Not enough matches: requires {self.min_match_count}, found {len(good_matches)}",
                "good_matches_count": len(good_matches)
            }

        # Extract matched points
        src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.last_query_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Save for debugging
        self.last_src_pts = src_pts
        self.last_dst_pts = dst_pts

        # Calculate homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            5.0,
            maxIters=2000,
            confidence=0.995
        )

        if homography is None:
            return {"success": False, "message": "Failed to compute homography"}

        # Transform template ROI corners to query image
        if self.template_roi_corners is not None:
            roi_corners = cv2.perspectiveTransform(
                self.template_roi_corners.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
        else:
            # Use entire template corners if ROI corners not available
            h, w = self.template.shape
            template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            roi_corners = cv2.perspectiveTransform(
                template_corners.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)

        # Calculate transformed ROI mask
        transformed_roi = cv2.warpPerspective(
            self.template_roi, homography, (query_img.shape[1], query_img.shape[0])
        )

        # Extract precise contour
        contours, _ = cv2.findContours(transformed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"success": False, "message": "Failed to extract ROI contour"}

        # Select largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Polygon approximation
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        return {
            "success": True,
            "homography": homography,
            "roi_corners": roi_corners,
            "polygon": polygon.reshape(-1, 2),
            "contour": largest_contour.reshape(-1, 2),
            "transformed_roi_mask": transformed_roi,
            "good_matches_count": len(good_matches),
            "inliers_count": np.sum(mask)
        }

    def visualize_result(self, query_image_path, detection_result, save_path=None):
        """
        Visualize detection result
        """
        query_img_color = cv2.imread(query_image_path)
        if query_img_color is None:
            return None

        if not detection_result["success"]:
            print(f"Detection failed: {detection_result['message']}")

            # Show match points distribution on failure
            if self.last_src_pts is not None and self.last_dst_pts is not None:
                plt.figure(figsize=(12, 6))

                # Template points
                plt.subplot(121)
                template_color = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
                for pt in self.last_src_pts:
                    cv2.circle(template_color, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)
                plt.imshow(cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB))
                plt.title(f"Template - {len(self.last_src_pts)} points")

                # Query points
                plt.subplot(122)
                query_color = cv2.cvtColor(self.last_query_img, cv2.COLOR_GRAY2BGR)
                for pt in self.last_dst_pts:
                    cv2.circle(query_color, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)
                plt.imshow(cv2.cvtColor(query_color, cv2.COLOR_BGR2RGB))
                plt.title(f"Query - {len(self.last_dst_pts)} points")

                plt.suptitle(f"Match Failed: {detection_result['message']}")
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path.replace('.jpg', '_points.jpg'))
                plt.show()

            return query_img_color

        # Draw ROI corners
        # roi_corners = detection_result["roi_corners"].astype(int)
        # cv2.polylines(query_img_color, [roi_corners], True, (0, 255, 0), 3)

        # Draw exact contour
        contour = detection_result["contour"].astype(int)
        cv2.polylines(query_img_color, [contour], True, (255, 0, 0), 2)

        # Draw polygon approximation
        # polygon = detection_result["polygon"].astype(int)
        # cv2.polylines(query_img_color, [polygon], True, (0, 0, 255), 2)

        # Add text info
        cv2.putText(query_img_color, f"Matches: {detection_result['good_matches_count']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(query_img_color, f"Inliers: {detection_result['inliers_count']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Legend
        cv2.putText(query_img_color, "Green: ROI Box", (10, query_img_color.shape[0] - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(query_img_color, "Blue: Exact Contour", (10, query_img_color.shape[0] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.putText(query_img_color, "Red: Polygon Approx", (10, query_img_color.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        if save_path:
            cv2.imwrite(save_path, query_img_color)

        return query_img_color

    def visualize_matches(self, detection_result, save_path=None):
        """
        Visualize feature matches (works for both success and failure)

        Args:
            detection_result: Result from detect_roi()
            save_path: Save path (optional)

        Returns:
            matplotlib figure
        """
        if self.template is None or self.last_query_img is None:
            print("No template or query image available")
            return None

        # Create color images for visualization
        template_color = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        query_color = cv2.cvtColor(self.last_query_img, cv2.COLOR_GRAY2BGR)

        # Draw all matches (including those filtered by Lowe's ratio test)
        all_matches_img = cv2.drawMatches(
            template_color, self.template_kp,
            query_color, self.last_query_kp,
            [m[0] for m in self.last_matches if len(m) >= 1], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Draw good matches (passed Lowe's ratio test)
        good_matches_img = cv2.drawMatches(
            template_color, self.template_kp,
            query_color, self.last_query_kp,
            self.last_good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),  # Green for good matches
            singlePointColor=(255, 0, 0)  # Blue for single points
        )

        # Create figure
        plt.figure(figsize=(16, 12))

        # All matches
        plt.subplot(211)
        plt.imshow(cv2.cvtColor(all_matches_img, cv2.COLOR_BGR2RGB))
        plt.title(f"All Matches ({len(self.last_matches)} points)")
        plt.axis('off')

        # Good matches
        plt.subplot(212)
        plt.imshow(cv2.cvtColor(good_matches_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Good Matches (after Lowe's ratio test: {len(self.last_good_matches)} points)")
        plt.axis('off')

        # Overall title
        status = "Success" if detection_result["success"] else "Failure"
        plt.suptitle(f"Feature Matching Result - {status}: {detection_result.get('message', '')}", fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return plt.gcf()


# Usage example
def main():
    # Create detector
    detector = SIFTROIDetector(min_match_count=10, ratio_threshold=0.8)

    # Load template and ROI mask
    template_path = "test/Image__2025-08-18__20-03-07.bmp"
    template_roi_path = "test/Image__2025-08-18__20-03-07_mask.png"
    # test_image_path = "test/Image__2025-08-18__20-04-17.bmp"
    test_image_path = "test/Image__2025-08-18__20-07-14.bmp"

    detector.load_template(template_path=template_path,
                           template_roi_path=template_roi_path)

    # Detect ROI
    result = detector.detect_roi(test_image_path)

    # Always visualize matches
    match_fig = detector.visualize_matches(result, "feature_matches.jpg")
    if match_fig:
        match_fig.show()  # Show match results

    if result["success"]:
        print("ROI detection successful!")
        print(f"Match points: {result['good_matches_count']}")
        print(f"Inliers: {result['inliers_count']}")
        print(f"Polygon vertices: {len(result['polygon'])}")
        print(f"Polygon coordinates:\n{result['polygon']}")

        # Visualize result
        vis_img = detector.visualize_result(test_image_path, result, "result.jpg")

        # Show result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title("SIFT ROI Detection Result")
        plt.axis('off')
        plt.show()

    else:
        print(f"ROI detection failed: {result['message']}")
        # Show point distribution on failure
        detector.visualize_result("test/Image__2025-08-17__17-02-08.bmp", result, "failure_debug.jpg")

# Batch processing example
def batch_detect_roi(detector, image_folder):
    """
    Batch process multiple images
    """
    import os
    import glob

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    results = {}
    for image_path in image_files:
        print(f"Processing: {image_path}")
        result = detector.detect_roi(image_path)
        results[image_path] = result

        # Always save match visualization
        match_save_path = image_path.replace('.jpg', '_matches.jpg').replace('.png', '_matches.png')
        detector.visualize_matches(result, match_save_path)

        if result["success"]:
            # Save result visualization
            output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
            detector.visualize_result(image_path, result, output_path)
        else:
            # Save failure debug
            debug_path = image_path.replace('.jpg', '_debug.jpg').replace('.png', '_debug.png')
            detector.visualize_result(image_path, result, debug_path)

    return results


if __name__ == "__main__":
    main()