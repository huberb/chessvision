from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import os
import chess
from pathlib import Path
import time


class Downloader():
    def __init__(self, output_dir="data"):
        options = Options()
        options.add_argument('--headless')
        self.browser = webdriver.Chrome(options=options)
        self.identifier_str = "<em>Find the best move for black.</em>"
        self.regex_pattern = r"\"pgn\":\".*?\""
        self.output_dir = output_dir
        Path(f"{output_dir}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/label").mkdir(parents=True, exist_ok=True)

    def save_new_game(self, index):
        self.browser.get("https://lichess.org/training")

        if self.identifier_str in self.browser.page_source:
            print("not from white's perspective..")
            return False

        if match := re.search(self.regex_pattern, self.browser.page_source):
            board = chess.Board()
            pgn = match.group(0)
            pgn = pgn.replace("\"pgn\":", "").replace("\"", "").split(" ")

            for move in pgn[:-1]:
                board.push_san(move)
            with open(f"{self.output_dir}/label/{index}.md", "w") as f:
                f.write(str(board))

            element = self.browser.find_element_by_tag_name("cg-board")
            element.screenshot(f"{self.output_dir}/images/{index}.png")
            print(board)
        return True

    def close(self):
        self.browser.close()


if __name__ == "__main__":
    downloader = Downloader()
    files = os.listdir("./data/images")
    index = max(int(file.split(".")[0]) for file in files)
    print(f"starting at index: {index}")
    while index < 10000:
        print(index)
        if downloader.save_new_game(index):
            index += 1
        time.sleep(1)
