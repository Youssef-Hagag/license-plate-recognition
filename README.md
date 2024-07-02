# Egyptian license Plate Recognition

The purpose of this project is to develop a robust and accurate Car License Plate Recognition (LPR) system. The system will be designed to automatically detect and recognize license plates from images, providing valuable applications in security, law enforcement, parking management, and more, we will mainly focus on the parking management application for this project.

## Table of Contents
- [Libraries Used](#libraries-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Libraries Used

List the libraries used in your project and their purposes:

- **NumPy**: Used for numerical operations and array manipulation in Python.
- **OpenCV (cv2)**: Employed for computer vision tasks like image processing, video capture, and manipulation.
- **Math**: Standard Python library for mathematical operations.
- **scikit-learn (sklearn)**: Used for machine learning tasks, including the K-Nearest Neighbors (KNN) classifier.
- **Numba**: Utilized for JIT (Just-In-Time) compilation to optimize performance in specific functions.
- **scikit-image**: Leveraged for image processing tasks, including contour finding and drawing shapes on images.
- **PySimpleGUI**: Employed for creating a graphical user interface (GUI) for the application.
- **PIL (Python Imaging Library)**: Used for opening, manipulating, and saving various image file formats.
- **Other libraries used in your project**

## Installation

Ensure you have Python installed on your system. Then, run the following commands in your terminal to install the required libraries:

```bash
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install numba
pip install scikit-image
pip install PySimpleGUI
```

## Usage

### Running the Application

To run the application:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Ensure you have Python installed.
4. Install the necessary dependencies (as mentioned in the Installation section).
5. Run the main script to start the application.

```bash
python UI.py
```

## Using the Application

Upon launching the application, follow these steps:

1. **Launch the Application**: Execute `UI.py` to start the application.
2. **Image Selection**: The GUI will appear, presenting options to select an image.
3. **Select Image**: Click on the 'Browse' button to choose an image file containing a vehicle and its license plate.
4. **Image Processing**: The application will process the image, perform license plate recognition, and display the result.
5. **Result Display**: The recognized license plate characters will be presented in the 'Processing Result' section of the GUI.
6. **LED Indicator**: An LED indicator will show 'Allowed' or 'Banned' based on a predefined comparison array.

## Contributing

Contributions to this project are highly appreciated. To contribute:

1. **Fork the Repository**: Fork this repository on GitHub.
2. **Create a New Branch**: Create a new branch for your feature or bug fix.
3. **Make Changes**: Implement your changes and ensure they are properly tested.
4. **Commit Changes**: Commit your modifications with clear and descriptive commit messages.
5. **Push Changes**: Push your changes to your forked repository.
6. **Open a Pull Request**: Submit a pull request to the `main` branch of this repository for review.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute it under the terms of this license.
