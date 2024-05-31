import os


root_dir = "/data2/yixiao/test_data"

dataset = 'slakh_allstems'
operation = 'extract'
model = 'AUDIT'

prediction_dir = f"{root_dir}/{dataset}/{operation}/{model}/output/"
ground_truth_dir = f"{root_dir}/{dataset}/{operation}/ground_truth/"
instruction_dir = f"{root_dir}/{dataset}/{operation}/instruction/"

files = [i for i in os.listdir(instruction_dir) if i.endswith(".txt")]

with open(os.path.join(instruction_dir, "text.csv"), "w") as f:
    f.write("filename,caption\n")
    for file in files:
        with open(os.path.join(instruction_dir, file), "r") as g:
            print(file)
            texts = g.readlines()
            ###
            # Instruction: Music piece.Instruct: Only Drums.
            # Stems: Drums, Bass, Piano, Guitar
            ###
            text_1 = texts[0].split(":")[-1].strip().replace('.', '')
            text_2 = texts[1].split(":")[-1].strip().replace(',', '')

            audio_name = file.replace(".txt", ".wav")

            if operation == "add":
                text_1 = text_1.replace("Add", "")
                text = f"{file},{text_1} {text_2} music"

            elif operation == "remove":
                text_1 = text_1.replace("No", "")
                text_2 = text_2.replace(text_1, "")
                text = f"{file},{text_1} music"

            elif operation == "extract":
                text_1 = text_1.replace("Only", "")
                text = f"{file},{text_1} music"

            print(text)
            f.write(text + "\n")

print("Finish writing text.csv")
