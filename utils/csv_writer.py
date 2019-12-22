import csv


def create_csv(filename, string, l):
    with open("New Results/" + filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        writer.writerow([string])
        writer.writerow([l])


def create_csv_list(filename, string, l):
    with open("New Results/" + filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        writer.writerow([string])
        for element in sorted(l):
            writer.writerow([element])


def create_csv_dictionary(filename, string, string1, string2, d):
    """Create csv files and arrange the contents in dictionary format"""
    with open("New Results/" + filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        writer.writerow([string])
        writer.writerow([string1, string2])
        for key, value in sorted(d.items()):
            writer.writerow([key, value])

