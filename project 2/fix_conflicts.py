import re
with open('sml_project_2.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r'<<<<<<< HEAD\r?\n\s*"execution_count": [0-9null]+,\r?\n=======\r?\n\s*"execution_count": [0-9null]+,\r?\n>>>>>>> [a-f0-9]+\r?\n')
new_text, num = pattern.subn(r'      "execution_count": null,\n', text)
print(f'Replaced {num} execution_count conflicts')

with open('sml_project_2.ipynb', 'w', encoding='utf-8') as f:
    f.write(new_text)
