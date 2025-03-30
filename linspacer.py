import torch


def get_points(low, up, target_size, group_size):
    if group_size == 1:
        split = (up - low)/(2*target_size)
        start = low + split
        end = up - split
        return torch.linspace(start, end, target_size)

    anchors = torch.linspace(low, up, target_size+1)
    all_points = []
    for i in range(target_size):
        points = torch.linspace(anchors[i], anchors[i+1], group_size)
        for p in points:
            all_points.append(p)

    all_points = torch.stack(all_points)
    return all_points


if __name__ == "__main__":
    points = get_points(0,100,10,3).tolist()
    print(points)
    points = get_points(0,100,10,2).tolist()
    print(points)
    points = get_points(50,100,10,2).tolist()
    print(points)
    points = get_points(0,100,10,1).tolist()
    print(points)
    points = get_points(50,100,10,1).tolist()
    print(points)