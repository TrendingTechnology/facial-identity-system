import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5 import QtWidgets


class CustomItem(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pass_info = QtWidgets.QLabel()
        self.time_info = QtWidgets.QLabel()
        self.train_btn = QtWidgets.QPushButton()
        self.train_btn.setText("🔼")
        self.train_btn.clicked.connect(self.train_model)
        self.setStyleSheet(
            """
            QLabel {
                margin: 0;
                padding: 0;
            }
            
            QPushButton {
                margin: 0;
                padding: 0;
                border: None
            }
            QPushButton::hover {
                font-size: 25px;
            }
            """
        )

        side_v_box = QtWidgets.QVBoxLayout()
        side_v_box.addWidget(self.pass_info)
        side_v_box.addWidget(self.time_info)

        main_h_box = QtWidgets.QHBoxLayout()
        main_h_box.addLayout(side_v_box, 1)
        main_h_box.addWidget(self.train_btn)

        self.setLayout(main_h_box)

    def set_pass_info(self, status, name=None):
        if status:
            self.pass_info.setText(f"🟢 {name}")
            self.train_btn.setToolTip("Select for training")
        else:
            self.pass_info.setText("🔴")
            self.train_btn.setDisabled(True)
            self.train_btn.setToolTip("Select for training (Disabled)")

    def set_time_info(self, time):
        self.time_info.setText(time)

    def train_model(self):
        # Obtain the model from parent->parent ....
        # I know the code its weird, but this it how it works...
        model = self.parent().parent().parent().parent().parent().video_container.model
        transform = self.parent().parent().parent().parent().parent().video_container.transform
        pos_lambda_ = self.parent().parent().parent().pos_lambda_
        neg_lambda_ = self.parent().parent().parent().neg_lambda_
        optimizer = self.parent().parent().parent().optimizer

        image = self.pix_to_array(self.parent().parent().parent().monitor.pixmap())

        if image is not None:
            user_id = self.parent().parent().parent().user_id
            model.train()
            target_features = self.parent().parent().parent().database.get(
                f"/users/{user_id}/identity", self.pass_info.text().split(" ", 1)[-1]
            )
            target_features = torch.FloatTensor(target_features).unsqueeze(0).repeat(2, 1)
            image = transform(image[..., :3].copy())
            pair_dis = nn.PairwiseDistance()

            # Create other positive features
            horizontal_flip_image = transforms.RandomHorizontalFlip(p=1.0)(image)
            positive_tensor = torch.stack([image, horizontal_flip_image], dim=0)

            # Create negative features
            vertical_flip_image = transforms.RandomVerticalFlip(p=1.0)(image)
            vertical_horizontal_image = transforms.RandomHorizontalFlip(p=1.0)(vertical_flip_image)
            negative_tensor = torch.stack([vertical_flip_image, vertical_horizontal_image], dim=0)

            pos_features = model.get_features(positive_tensor)
            neg_features = model.get_features(negative_tensor)

            loss = pair_dis(target_features, pos_features) + (1 - pair_dis(target_features, neg_features))

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # Change back to eval mode
            model.eval()

            torch.save(model.state_dict(), "weight/model.pt")

            bucket = self.parent().parent().parent().bucket
            model_weight_blob = bucket.blob("model.pt")
            model_weight_blob.upload_from_filename("weight/model.pt")
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("The image is not available")
            msg.setWindowTitle("Error for doing training")
            msg.exec_()

    @staticmethod
    def pix_to_array(pixmap):
        if pixmap is not None:
            h = pixmap.size().height()
            w = pixmap.size().width()

            q_image = pixmap.toImage()
            byte_str = q_image.bits().asstring(w * h * 4)

            img = np.frombuffer(byte_str, dtype=np.uint8).reshape((h, w, 4))
            return img

        return None
