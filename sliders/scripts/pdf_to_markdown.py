import argparse
import concurrent.futures
import glob
import os

from docling.document_converter import ConversionStatus, DocumentConverter
from loguru import logger
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a PDF file to a string")
    parser.add_argument(
        "--pdf_dir", type=str, help="Path to the directory containing the PDF files"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to the directory to save the output files"
    )
    parser.add_argument(
        "--split_pages",
        help="Whether to split the pages of the PDF file",
        action="store_true",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for the conversion",
        default=4,
    )
    return parser


def split_and_convert_pdf(file_name, file, output_file, converter):
    temp_first = f"temp_first_{os.getpid()}.pdf"
    temp_rest = f"temp_rest_{os.getpid()}.pdf"

    try:
        with open(file, "rb") as f:
            inputpdf = PdfReader(f)
            # First page
            output = PdfWriter()
            output.add_page(inputpdf.pages[0])
            with open(temp_first, "wb") as outputStream:
                output.write(outputStream)

            result = converter.convert(temp_first)
            md_result = result.document.export_to_markdown()
            with open(output_file, "w") as f:
                f.write(md_result)

            # Rest if the pages
            output = PdfWriter()
            for i in range(1, len(inputpdf.pages)):
                output.add_page(inputpdf.pages[i])
            with open(temp_rest, "wb") as outputStream:
                output.write(outputStream)

            result = converter.convert(temp_rest)
            md_result = result.document.export_to_markdown()
            with open(output_file, "a") as f:
                f.write(md_result)
    finally:
        # Clean up temporary files
        for temp_file in [temp_first, temp_rest]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def convert_from_pdf(file, converter, output_dir, split_pages: bool = False):
    try:
        if file.endswith(".pdf"):
            # Seperate into two pdfs. first page and then later pages.
            file_name = os.path.basename(file)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, file_name.replace(".pdf", ".md"))

            # if file exists, skip
            if os.path.exists(output_file):
                logger.warning(f"Skipping {file} because it already exists")
                return

            if split_pages:
                split_and_convert_pdf(file, output_dir, converter)
            else:
                result = converter.convert(file)
                md_result = result.document.export_to_markdown()
                with open(output_file, "w") as f:
                    f.write(md_result)

        else:
            logger.warning(f"Skipping {file} because it is not a PDF file")

        return True
    except Exception as e:
        logger.error(f"Error converting file: {file} : {e}")
        return False


def convert_from_pdfs(files, output_dir, converter, split_pages: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    results = converter.convert_all(files)
    for result in results:
        output_file = os.path.join(
            output_dir, result.input.file.name.replace(".pdf", ".md")
        )
        if result.status == ConversionStatus.SUCCESS:
            md_result = result.document.export_to_markdown()
            with open(output_file, "w") as f:
                f.write(md_result)
        else:
            logger.warning(
                f"Skipping {result.input.file.name} because it failed to convert"
            )


def convert_from_pdf_in_dir(
    pdf_dir: str, output_dir: str, split_pages: bool = False, num_workers: int = 4
) -> None:
    """Convert all the PDF files in the given directory to text files using multiple threads"""
    os.makedirs(output_dir, exist_ok=True)

    converter = DocumentConverter()
    files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    failed_files = []
    successful_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                convert_from_pdf, file, converter, output_dir, split_pages
            ): file
            for file in files
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_file), total=len(files)
        ):
            file = future_to_file[future]
            try:
                future.result()
                successful_files.append(file)
            except Exception as e:
                logger.error(f"Error converting file '{file}': {e}")
                failed_files.append((file, str(e)))
    # Summary report
    logger.info(f"Successfully converted {len(successful_files)} files")
    if failed_files:
        logger.warning(f"Failed to convert {len(failed_files)} files:")
        for file, error in failed_files:
            logger.warning(f"  - {file}: {error}")

    # first failed file
    if failed_files:
        logger.error(f"First failed file: {failed_files[0][0]}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    convert_from_pdf_in_dir(
        args.pdf_dir, args.output_dir, args.split_pages, args.num_workers
    )