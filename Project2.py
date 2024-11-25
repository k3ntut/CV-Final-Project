import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PATH = 'dataset/dataset_updated/training_set/'
PATH_VALIDATION = 'dataset/dataset_updated/validation_set/'
categories = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

def extract_features(folder_path, dataset, label):
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()

    for image_name in os.listdir(folder_path):
        image_path = f"{folder_path}/{image_name}"
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobelx, sobely).astype(np.uint8)

        sift_keypoints, sift_descriptors = sift.detectAndCompute(sobel_edges, None)
        orb_keypoints, orb_descriptors = orb.detectAndCompute(sobel_edges, None)

        if sift_descriptors is not None and orb_descriptors is not None:
            sift_mean = np.mean(sift_descriptors, axis=0)
            orb_mean = np.mean(orb_descriptors, axis=0)
            combined_features = np.hstack((sift_mean, orb_mean))
            dataset.append((combined_features, label, image))

def match_keypoints(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

train_features = []
val_features = []

for category in categories:
    folder_path = f"{PATH}/{category}"
    extract_features(folder_path, train_features, category)

for category in categories:
    folder_path = f"{PATH_VALIDATION}/{category}"
    extract_features(folder_path, val_features, category)

X_train = [item[0] for item in train_features]
y_train = [item[1] for item in train_features]

X_val = [item[0] for item in val_features]
y_val = [item[1] for item in val_features]
images_val = [item[2] for item in val_features]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_encoded)

y_pred = clf.predict(X_val)

print("Validation Classification Report:")
print(classification_report(y_val_encoded, y_pred, target_names=le.classes_))

accuracy = accuracy_score(y_val_encoded, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

def display_all_validation_results(images, y_val, y_pred, le, items_per_page=10):
    total_images = len(images)
    pages = (total_images + items_per_page - 1) // items_per_page

    for page in range(pages):
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, total_images)

        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(range(start_idx, end_idx)):
            plt.subplot(2, items_per_page // 2, i + 1) 
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            true_label = le.inverse_transform([y_val[idx]])[0]
            predicted_label = le.inverse_transform([y_pred[idx]])[0]
            plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.tight_layout()
        plt.show()

display_all_validation_results(images_val, y_val_encoded, y_pred, le, items_per_page=10)

def visualize_keypoint_matches(image1, image2, descriptors1, descriptors2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    matches = match_keypoints(descriptors1, descriptors2)

    match_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Keypoints Matching")
    plt.show()