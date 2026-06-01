import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import csv
import time


class DataCollector(Node):
    def __init__(self):
        super().__init__('lidar_data_collector')

        self.subscription = self.create_subscription(LaserScan, '/scan', self.listener_callback, 10)

        self.csv_file = open('raw_lidar_data.csv', mode = 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.get_logger().info('Saving to .csv file')
        self.get_logger().info('Ctrl+C to stop execution')

    def listener_callback(self,msg):

        laser_points = list(msg.ranges)

        self.csv_writer.writerow(laser_points)
        self.get_logger().info('Row saved...')
        time.sleep(0.3)

    
def main(args=None):

    rclpy.init(args=args)
    lidar_collector = DataCollector()

    try:
        rclpy.spin(lidar_collector)
    except KeyboardInterrupt:
        lidar_collector.get_logger().info('Data collected..')
    finally:
        lidar_collector.csv_file.close()
        lidar_collector.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
    