import sys


def main():
    if len(sys.argv) < 3:
        print("\033[91mUsage: py fix_input.py <agent-to-replace> <student-ID>\033[0m")
        exit(1)

    agent_to_replace, student_ID = sys.argv[1], sys.argv[2]

    # edit STcpClient.py
    with open("STcpClient.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("STcpClient.py", "w", encoding="utf-8") as f:
        f.writelines([f"idTeam = {agent_to_replace}\n" if "idTeam = " in line else line for line in lines])

    # edit input.txt
    with open(f"input.txt", "w") as f:
        for agent in range(1, 5):
            f.write(f'{agent}\n')
            f.write(f"{student_ID}.exe\n" if agent == int(agent_to_replace) else f"./sample/Sample_{agent}.exe\n")
        f.write(f'{agent_to_replace}\n')


if __name__ == "__main__":
    main()
