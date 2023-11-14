disks = int(input("Enter number of disks: "))
def move(n, start, finish, temp, counter):
    if n > 0:
        counter[0] += 1
        move(n - 1, start, temp, finish, counter)
        print(f"Move disk {n} from tower {start} to tower {finish}")
        move(n - 1, temp, finish, start, counter)
def main():
    counter = [0]
    move(disks, 1, 3, 2, counter)
    print(f"Total moves: {counter[0]}")
if __name__ == "__main__":
    main()