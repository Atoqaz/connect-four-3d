place_in_slot = -1
while place_in_slot not in [7,8,9]:

    try:
        print(
            f"Turn = Player: "
        )
        place_in_slot = int(
            input(
                f"Select slot for the piece:: "
            )
        )
    except ValueError:
        continue

    if place_in_slot in [7,8,9]:
        break
    else:
        print("Incorrect input")