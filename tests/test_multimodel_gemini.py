import base64
import os
from unittest.mock import Mock, patch

import pytest

from app.multimodel_gemini import generate_image


@pytest.fixture
def mock_image_base64():
    png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    return base64.b64encode(png_data).decode("utf-8")


@pytest.fixture
def mock_response(mock_image_base64):
    response = Mock()
    data_url = f"data:image/png;base64,{mock_image_base64}"
    response.content = [{"image_url": {"url": data_url}}]
    return response


@patch("app.multimodel_gemini.ChatGoogleGenerativeAI")
def test_generate_image_writes_file(mock_model_class, mock_response, tmp_path):
    mock_model_instance = Mock()
    mock_model_instance.invoke.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        output_path = generate_image()

        assert output_path == "./generated_cat.png"
        assert os.path.exists(output_path)

        with open(output_path, "rb") as f:
            file_content = f.read()

        assert file_content.startswith(b"\x89PNG")
        assert len(file_content) > 0
    finally:
        os.chdir(original_dir)


@patch("app.multimodel_gemini.ChatGoogleGenerativeAI")
def test_generate_image_correct_decoding(
    mock_model_class, mock_response, mock_image_base64, tmp_path
):
    mock_model_instance = Mock()
    mock_model_instance.invoke.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        generate_image()

        with open("./generated_cat.png", "rb") as f:
            decoded_content = f.read()

        expected_bytes = base64.b64decode(mock_image_base64)
        assert decoded_content == expected_bytes
    finally:
        os.chdir(original_dir)
