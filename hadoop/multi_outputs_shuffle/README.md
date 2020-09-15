# Usage

sh auto_submit_job.sh date_start section_index(0-based) [date_last]

# Description

We have to specify a hour section index to trigger the automatical shuffling jobs. And these sequential jobs start with "date_start" and end with "date_last".

E.g. sh auto_submit_job.sh 20200604 0 20201231

The total 24 hours are split to 5 hour sections with their index mapping are:

|hour section|index|
|------------|-----|
|  [0, 3]    |  0  |
|  [4, 7]    |  1  |
|  [8, 12]   |  2  |
|  [13, 18]  |  3  |
|  [19, 23]  |  4  |

