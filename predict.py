import json


def load_params(filename: str) -> tuple[float, float]:
    try:
        with open(filename, "r") as file:
            content = file.read().strip()
            if not content:
                raise ValueError("File is empty.")
            data = json.loads(content)
            theta0 = data.get("theta0")
            theta1 = data.get("theta1")
            if theta0 is None or theta1 is None:
                raise ValueError("Missing 'theta0' or 'theta1' in JSON file.")
            return theta0, theta1
    except Exception as e:
        raise RuntimeError(f"Error while loading thetas: {e}")


def predict(tetha0, tetha1, mileage) -> int:
    return tetha0 + tetha1 * mileage

if __name__ == "__main__":
    tetha0, tetha1 = load_params("model_params.json")
    while True:
        try:
            mileage = int(input("Please enter your mileage: "))
            print(f"Estimated price: {max(0, round(predict(tetha0, tetha1, mileage)))} euros.")
            exit(0)
        except Exception as e:
            print(f"Wrong input: {e}")
