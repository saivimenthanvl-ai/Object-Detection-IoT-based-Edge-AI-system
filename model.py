import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import tensorflow_models as tfm
    from official.core import exp_factory
    from official.vision.serving import export_saved_model_lib
except ImportError:
    tfm = None
    exp_factory = None
    export_saved_model_lib = None


# =========================================================
# 1. PROJECT PATHS
# =========================================================
PROJECT_DIR = Path(r"Object Detection IoT-based Edge AI system")
DATA_DIR = PROJECT_DIR / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"

TRAIN_RECORD = str(DATA_DIR / "train.tfrecord")
VAL_RECORD = str(DATA_DIR / "val.tfrecord")

LABEL_MAP_PATH = str(DATA_DIR / "label_map.pbtxt")
MODEL_DIR = str(PROJECT_DIR / "trained_model")
EXPORT_DIR = str(PROJECT_DIR / "exported_model")
TFLITE_DIR = str(PROJECT_DIR / "tflite")

for p in [
    PROJECT_DIR,
    DATA_DIR,
    ANNOTATIONS_DIR,
    IMAGES_DIR,
    Path(MODEL_DIR),
    Path(EXPORT_DIR),
    Path(TFLITE_DIR),
]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. CONFIG
# =========================================================
NUM_CLASSES = 3
IMAGE_SIZE = 640
BATCH_SIZE = 2
TRAIN_STEPS = 100
VAL_STEPS = 10
NUM_SYNTHETIC_IMAGES = 12

CLASS_NAMES = ["class_1", "class_2", "class_3"]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i + 1: name for i, name in enumerate(CLASS_NAMES)}

assert len(CLASS_NAMES) == NUM_CLASSES
print("CLASS_TO_ID =", CLASS_TO_ID)


# =========================================================
# 3. WRITE LABEL MAP
# =========================================================
def write_label_map(label_map_path: str, class_names: list[str]) -> None:
    with open(label_map_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names, start=1):
            f.write("item {\n")
            f.write(f"  id: {i}\n")
            f.write(f"  name: '{name}'\n")
            f.write("}\n\n")


write_label_map(LABEL_MAP_PATH, CLASS_NAMES)
print("Saved label map:", LABEL_MAP_PATH)


# =========================================================
# 4. GENERATE SYNTHETIC IMAGES + ANNOTATIONS
# =========================================================
def generate_synthetic_dataset(images_dir: Path, annotation_json_path: Path, num_images: int = 12) -> None:
    random.seed(42)
    annotations = []

    width, height = 640, 480

    for idx in range(1, num_images + 1):
        image_name = f"img{idx:03d}.jpg"
        image_path = images_dir / image_name

        # white background
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        objects = []
        num_objects = random.randint(1, 3)

        for _ in range(num_objects):
            label = random.choice(CLASS_NAMES)

            xmin = random.randint(20, width - 180)
            ymin = random.randint(20, height - 180)
            xmax = xmin + random.randint(60, 150)
            ymax = ymin + random.randint(60, 150)

            xmax = min(xmax, width - 10)
            ymax = min(ymax, height - 10)

            # simple colored boxes to simulate different classes
            if label == "class_1":
                color = (255, 0, 0)
            elif label == "class_2":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

            objects.append({
                "label": label,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

        image.save(image_path, quality=95)

        annotations.append({
            "image": image_name,
            "width": width,
            "height": height,
            "objects": objects
        })

    with open(annotation_json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print("Generated synthetic dataset at:", images_dir)
    print("Saved annotations at:", annotation_json_path)


ANNOTATION_JSON = ANNOTATIONS_DIR / "annotations.json"

# only generate if missing
if not ANNOTATION_JSON.exists():
    generate_synthetic_dataset(IMAGES_DIR, ANNOTATION_JSON, NUM_SYNTHETIC_IMAGES)


# =========================================================
# 5. LOAD ANNOTATIONS
# =========================================================
with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
    annotations = json.load(f)

print("Total images:", len(annotations))
print("Sample annotation:", annotations[0])


# =========================================================
# 6. TRAIN / VAL SPLIT
# =========================================================
random.seed(42)
random.shuffle(annotations)

split_idx = int(0.9 * len(annotations))
train_annotations = annotations[:split_idx]
val_annotations = annotations[split_idx:]

print("Train images:", len(train_annotations))
print("Val images:", len(val_annotations))


# =========================================================
# 7. TFRECORD HELPERS
# =========================================================
def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# =========================================================
# 8. CREATE TF EXAMPLE
# =========================================================
def create_tf_example(sample, images_dir: Path, class_to_id: dict):
    image_path = images_dir / sample["image"]

    with tf.io.gfile.GFile(str(image_path), "rb") as fid:
        encoded_jpg = fid.read()

    image = Image.open(image_path)
    width, height = image.size
    filename = sample["image"].encode("utf-8")
    image_format = image_path.suffix.lower().replace(".", "").encode("utf-8")

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for obj in sample["objects"]:
        xmin = obj["xmin"] / width
        xmax = obj["xmax"] / width
        ymin = obj["ymin"] / height
        ymax = obj["ymax"] / height

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        classes_text.append(obj["label"].encode("utf-8"))
        classes.append(class_to_id[obj["label"]])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(filename),
        "image/source_id": bytes_feature(filename),
        "image/encoded": bytes_feature(encoded_jpg),
        "image/format": bytes_feature(image_format),
        "image/object/bbox/xmin": float_list_feature(xmins),
        "image/object/bbox/xmax": float_list_feature(xmaxs),
        "image/object/bbox/ymin": float_list_feature(ymins),
        "image/object/bbox/ymax": float_list_feature(ymaxs),
        "image/object/class/text": bytes_list_feature(classes_text),
        "image/object/class/label": int64_list_feature(classes),
    }))
    return tf_example


# =========================================================
# 9. WRITE TFRECORD
# =========================================================
def write_tfrecord(samples, output_path: str, images_dir: Path, class_to_id: dict):
    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in samples:
            tf_example = create_tf_example(sample, images_dir, class_to_id)
            writer.write(tf_example.SerializeToString())


write_tfrecord(train_annotations, TRAIN_RECORD, IMAGES_DIR, CLASS_TO_ID)
write_tfrecord(val_annotations, VAL_RECORD, IMAGES_DIR, CLASS_TO_ID)

print("Saved:", TRAIN_RECORD)
print("Saved:", VAL_RECORD)


# =========================================================
# 10. SHOW SAMPLE IMAGE
# =========================================================
def show_sample(sample, images_dir: Path):
    image_path = images_dir / sample["image"]
    image = np.array(Image.open(image_path).convert("RGB"))

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()

    for obj in sample["objects"]:
        xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        w, h = xmax - xmin, ymax - ymin
        rect = Rectangle((xmin, ymin), w, h, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(0, ymin - 5),
            obj["label"],
            fontsize=10,
            bbox=dict(facecolor="yellow", alpha=0.5)
        )

    plt.axis("off")
    plt.show()


show_sample(train_annotations[0], IMAGES_DIR)


# =========================================================
# 11. MODEL GARDEN SETUP
# =========================================================
if tfm is None or exp_factory is None or export_saved_model_lib is None:
    print("\nTensorFlow Model Garden packages are not installed.")
    print("Install with:")
    print("pip install tf-models-official pycocotools pillow matplotlib")
else:
    exp_config = exp_factory.get_exp_config("retinanet_resnetfpn_coco")

    exp_config.task.model.num_classes = NUM_CLASSES + 1
    exp_config.task.model.input_size = [IMAGE_SIZE, IMAGE_SIZE, 3]

    exp_config.task.train_data.input_path = TRAIN_RECORD
    exp_config.task.train_data.dtype = "float32"
    exp_config.task.train_data.global_batch_size = BATCH_SIZE
    exp_config.task.train_data.is_training = True

    exp_config.task.validation_data.input_path = VAL_RECORD
    exp_config.task.validation_data.dtype = "float32"
    exp_config.task.validation_data.global_batch_size = BATCH_SIZE
    exp_config.task.validation_data.is_training = False

    exp_config.trainer.train_steps = TRAIN_STEPS
    exp_config.trainer.validation_steps = VAL_STEPS
    exp_config.trainer.validation_interval = 20
    exp_config.trainer.steps_per_loop = 20
    exp_config.trainer.summary_interval = 20
    exp_config.trainer.checkpoint_interval = 20

    exp_config.task.annotation_file = None

    if tf.config.list_physical_devices("GPU"):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    print("Replicas:", strategy.num_replicas_in_sync)

    task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=MODEL_DIR)

    with strategy.scope():
        model = task.build_model()
        task.initialize(model)

    print("Model built successfully.")

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=strategy,
        task=task,
        mode="train_and_eval",
        params=exp_config,
        model_dir=MODEL_DIR,
        run_post_eval=True
    )

    print("Final eval logs:")
    print(eval_logs)

    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=1,
        input_image_size=[IMAGE_SIZE, IMAGE_SIZE],
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(MODEL_DIR),
        export_dir=EXPORT_DIR
    )

    print("Exported SavedModel to:", EXPORT_DIR)

    imported = tf.saved_model.load(EXPORT_DIR)
    infer = imported.signatures["serving_default"]
    print(infer.structured_input_signature)
    print(infer.structured_outputs)

    def load_image_np(path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    TEST_IMAGE = str(IMAGES_DIR / train_annotations[0]["image"])

    image_np = load_image_np(TEST_IMAGE)
    input_tensor = tf.convert_to_tensor(image_np[None, ...], dtype=tf.uint8)

    outputs = infer(input_tensor)

    for k, v in outputs.items():
        print(k, v.shape)

    def draw_predictions(image_np, outputs, id_to_class, score_threshold=0.3):
        boxes = outputs["detection_boxes"][0].numpy()
        classes = outputs["detection_classes"][0].numpy().astype(int)
        scores = outputs["detection_scores"][0].numpy()

        h, w, _ = image_np.shape
        img = image_np.copy()

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        ax = plt.gca()

        for box, cls_id, score in zip(boxes, classes, scores):
            if score < score_threshold:
                continue

            ymin, xmin, ymax, xmax = box
            xmin, xmax = xmin * w, xmax * w
            ymin, ymax = ymin * h, ymax * h

            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="blue",
                linewidth=2
            )
            ax.add_patch(rect)
            label = f"{id_to_class.get(cls_id, cls_id)}: {score:.2f}"
            ax.text(
                xmin,
                max(0, ymin - 5),
                label,
                fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.5)
            )

        plt.axis("off")
        plt.show()

    draw_predictions(image_np, outputs, ID_TO_CLASS, score_threshold=0.3)

    FLOAT_TFLITE_PATH = str(Path(TFLITE_DIR) / "detector_float.tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
    tflite_model = converter.convert()

    with open(FLOAT_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("Saved float TFLite model:", FLOAT_TFLITE_PATH)

    DYNAMIC_TFLITE_PATH = str(Path(TFLITE_DIR) / "detector_dynamic.tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()

    with open(DYNAMIC_TFLITE_PATH, "wb") as f:
        f.write(tflite_dynamic)

    print("Saved dynamic-range quantized TFLite model:", DYNAMIC_TFLITE_PATH)

    def representative_dataset():
        sample_count = min(100, len(train_annotations))
        for i in range(sample_count):
            img_path = IMAGES_DIR / train_annotations[i]["image"]
            img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img).astype(np.float32)
            img = np.expand_dims(img, axis=0)
            yield [img]

    INT8_TFLITE_PATH = str(Path(TFLITE_DIR) / "detector_int8.tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_int8 = converter.convert()
        with open(INT8_TFLITE_PATH, "wb") as f:
            f.write(tflite_int8)
        print("Saved INT8 TFLite model:", INT8_TFLITE_PATH)
    except Exception as e:
        print("INT8 conversion failed:")
        print(e)

    TFLITE_MODEL_TO_TEST = DYNAMIC_TFLITE_PATH

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_TO_TEST)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:", input_details)
    print("Output details:", output_details)

    img = Image.open(TEST_IMAGE).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)

    input_info = input_details[0]
    input_dtype = input_info["dtype"]

    input_data = np.expand_dims(img, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    elif input_dtype == np.int8:
        scale, zero_point = input_info["quantization"]
        input_data = input_data.astype(np.float32) / scale + zero_point
        input_data = input_data.astype(np.int8)
    elif input_dtype == np.uint8:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_info["index"], input_data)
    interpreter.invoke()

    tflite_outputs = []
    for out in output_details:
        tflite_outputs.append(interpreter.get_tensor(out["index"]))

    for i, out in enumerate(tflite_outputs):
        print(f"Output {i} shape:", out.shape)

    print("SavedModel:", EXPORT_DIR)
    print("Float TFLite:", FLOAT_TFLITE_PATH)
    print("Dynamic TFLite:", DYNAMIC_TFLITE_PATH)
    print("Training checkpoints:", MODEL_DIR)