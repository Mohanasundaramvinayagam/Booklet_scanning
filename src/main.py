from preprocessing import preprocess_image


INPUT_IMAGE = "data/input/handwrittten_char.jpeg"
OUTPUT_DIR = "data/output"


try:

    results = preprocess_image(
        INPUT_IMAGE,
        OUTPUT_DIR
    )

    print()
    print("================================")
    print("PREPROCESSING COMPLETED")
    print("================================")

except FileNotFoundError as error:

    print("❌", error)