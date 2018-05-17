import numpy as np
import sys


class FiltersCV:
    def __init__(self, img):
        """init filter class and create working copy

           Parameters
           ----------
           img : image - array with shape(height, width, color_dims)
        """

        self.img_copy = np.copy(img)

        self.width = img.shape[1]
        self.height = img.shape[0]

        self.pixel_default_value = np.array([0, 0])
        self.pixels = img
        self.pixels_copy = self.img_copy

    def get_index(self, u, v):
        """return index of pixel, restrict non valid u, v combinations if u or v is not valid, returns border index

        Parameters
        ----------
        u : int
            height index of image

        v : int
            width index of image

         Returns
        ----------
        out : np.array(u, v)
            valid index of image
        """

        if u < 0:
            u = 0
        elif u >= self.height:
            u = self.height - 1

        if v < 0:
            v = 0
        elif v >= self.width:
            v = self.width - 1

        return np.array([u, v])

    def get_pixel_rgb(self, u, v, rgb):
        """return pixel rgb values

        Parameters
        ----------
        u : int
            height index of image

        v : int
            width index of image

        rgb : [r, g, b]
            out array
        """

        i = self.get_index(u, v)

        c = self.pixels[i[0], i[1]]

        rgb[0] = c[0]
        rgb[1] = c[1]
        rgb[2] = c[2]

    @staticmethod
    def clamp(val):
        """limit value between 0..255.

        Parameters
        ----------
        val : int
            editable value

        Returns
        ----------
        out : int
            valid value between 0..255
        """

        if val < 0:
            return 0
        if val > 255:
            return 255

        return int(val)

    def set_pix(self, u, v, value_f):
        """set pixel rgb values. Includes

        Parameters
        ----------
        u : int
            height index of image

        v : int
            width index of image

        value_f : [r, g, b]
            array of rgb values, which would be set to (u, v) pixel
        """

        rgb = [0, 0, 0]
        if 0 <= u < self.height and 0 <= v < self.width:
            if len(value_f) == 3:
                rgb[0] = self.clamp(np.round(value_f[0]))
                rgb[1] = self.clamp(np.round(value_f[1]))
                rgb[2] = self.clamp(np.round(value_f[2]))
            else:
                rgb[0] = self.clamp(np.round(value_f[0]))
                rgb[1] = rgb[0]
                rgb[2] = rgb[0]

            self.pixels_copy[u, v][0] = rgb[0]
            self.pixels_copy[u, v][1] = rgb[1]
            self.pixels_copy[u, v][2] = rgb[2]

    def filter_kuwara(self, radius, t_sigma):
        """kuwara image filter

        Parameters
        ----------
        radius : int
            filter radius

        t_sigma : float
            Threshold on sigma to avoid banding in flat regions

         Returns
        ----------
        out : img
            array with shape (height, width, 3)
        """

        r = radius

        # fixed sub_region size
        n = (r + 1) * (r + 1)
        dm = int(r / 2 - r)
        dp = dm + r

        s_min = sys.float_info.max
        a_min = sys.float_info.max
        a_min_r = sys.float_info.max
        a_min_g = sys.float_info.max
        a_min_b = sys.float_info.max

        def check_radius(radius_param):
            if radius_param < 1:
                raise ValueError('filter radius must me >= 1')

        check_radius(r)

        def filter_pixel(u, v):
            """provide (u, v) pixel filtering

            Parameters
            ----------
            u : int
                height index of image

            v : int
                width index of image

             Returns
            ----------
            out : [r, g, b]
                array of red, green and blue channel values
            """

            global s_min
            global a_min_r
            global a_min_g
            global a_min_b

            s_min = sys.float_info.max

            eval_sub_region(u + dm, v + dm)

            s_min = s_min - 3 * t_sigma * n

            eval_sub_region(u + dm, v + dm)
            eval_sub_region(u + dm, v + dp)
            eval_sub_region(u + dp, v + dm)
            eval_sub_region(u + dp, v + dp)

            rgb[0] = np.rint(a_min_r)
            rgb[1] = np.rint(a_min_g)
            rgb[2] = np.rint(a_min_b)

            return rgb

        def eval_sub_region(u, v):
            """evaluate the sub region centered at (u,v)

            Parameters
            ----------
            u : int
                height index of image

            v : int
                width index of image
            """

            global s_min
            global a_min_r
            global a_min_g
            global a_min_b

            c_pix = [0, 0, 0]

            s1_r, s2_r, s1_g, s2_g, s1_b, s2_b = 0, 0, 0, 0, 0, 0

            for j in range(dm, dp + 1):
                for i in range(dm, dp + 1):
                    self.get_pixel_rgb(u + i, v + j, c_pix)
                    red = c_pix[0]
                    grn = c_pix[1]
                    blu = c_pix[2]

                    s1_r = s1_r + red
                    s1_g = s1_g + grn
                    s1_b = s1_b + blu

                    s2_r = s2_r + red ** 2
                    s2_g = s2_g + grn ** 2
                    s2_b = s2_b + blu ** 2

            nf = n
            s_r = s2_r - s1_r * s1_r / nf
            s_g = s2_g - s1_g * s1_g / nf
            s_b = s2_b - s1_b * s1_b / nf

            s_rgb = s_r + s_g + s_b

            if s_rgb < s_min:
                s_min = s_rgb
                a_min_r = s1_r / n
                a_min_g = s1_g / n
                a_min_b = s1_b / n

        rgb = [0, 0, 0]
        h = self.height
        w = self.width

        for v in range(0, h):
            for u in range(0, w):
                rgb = filter_pixel(u, v)
                self.set_pix(u, v, rgb)

        img_result = self.pixels_copy.reshape(self.height, self.width, 3)

        return img_result

    def filter_nagao(self, variance_threshold):
        """Nagao Matsuyama image filter

        Parameters
        ----------
        variance_threshold : float 0..10
            filter radius

        Returns
        ----------
        out : img
            array with shape (height, width, 3)
        """
        var_threshold = variance_threshold

        min_variance = sys.float_info.max
        min_mean = sys.float_info.max
        min_mean_r = sys.float_info.max
        min_mean_g = sys.float_info.max
        min_mean_b = sys.float_info.max

        r1 = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

        r2 = [[-2, -1], [-1, -1], [-2, 0], [-1, 0], [0, 0], [-2, 1], [-1, 1]]

        r3 = [[-2, -2], [-1, -2], [-2, -1], [-1, -1], [0, -1], [-1, 0], [0, 0]]

        r4 = [[-1, -2], [0, -2], [1, -2], [-1, -1], [0, -1], [1, -1], [0, 0]]

        r5 = [[1, -2], [2, -2], [0, -1], [1, -1], [2, -1], [0, 0], [1, 0]]

        r6 = [[1, -1], [2, -1], [0, 0], [1, 0], [2, 0], [1, 1], [2, 1]]

        r7 = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2]]

        r8 = [[0, 0], [-1, 1], [0, 1], [1, 1], [-1, 2], [0, 2], [1, 2]]

        r9 = [[-1, 0], [0, 0], [-2, 1], [-1, 1], [0, 1], [-2, 2], [-1, 2]]

        sub_regions = [r2, r3, r4, r5, r6, r7, r8, r9]

        def filter_pixel_nagao(u, v):
            """provide (u, v) pixel filtering

            Parameters
            ----------
            u : int
                height index of image

            v : int
                width index of image

             Returns
            ----------
            out : [r, g, b]
                array of red, green and blue channel values
            """
            global min_variance
            global min_mean_r
            global min_mean_g
            global min_mean_b

            min_variance = sys.float_info.max

            eval_sub_region_color_nagao(r1, u, v)

            min_variance = min_variance - (3 * var_threshold)

            for Rk in sub_regions:
                eval_sub_region_color_nagao(Rk, u, v)

            rgb[0] = int(np.rint(min_mean_r))
            rgb[1] = int(np.rint(min_mean_g))
            rgb[2] = int(np.rint(min_mean_b))

            return rgb

        def eval_sub_region_color_nagao(r, u, v):
            """evaluate the subregion centered at (u,v)

            Parameters
            ----------
            u : int
                height index of image

            v : int
                width index of image
            """

            global min_variance
            global min_mean_r
            global min_mean_g
            global min_mean_b

            sum_1r, sum_2r, sum_1g, sum_2g, sum_1b, sum_2b = 0, 0, 0, 0, 0, 0

            n = 0
            for p in r:
                c_pix = [0, 0, 0]
                self.get_pixel_rgb(u + p[0], v + p[1], c_pix)
                red = int(c_pix[0])
                grn = int(c_pix[1])
                blu = int(c_pix[2])

                sum_1r = sum_1r + red
                sum_1g = sum_1g + grn
                sum_1b = sum_1b + blu
                sum_2r = sum_2r + red * red
                sum_2g = sum_2g + grn * grn
                sum_2b = sum_2b + blu * blu
                n = n + 1

            nr = n
            var_r = (sum_2r - sum_1r * sum_1r / nr) / nr
            var_g = (sum_2g - sum_1g * sum_1r / nr) / nr
            var_b = (sum_2b - sum_1b * sum_1b / nr) / nr

            total_var = var_r + var_g + var_b
            if total_var < min_variance:
                min_variance = total_var
                min_mean_r = sum_1r / nr
                min_mean_g = sum_1g / nr
                min_mean_b = sum_1b / nr

        rgb = [0, 0, 0]

        h = self.height
        w = self.width

        for v in range(0, h):
            for u in range(0, w):
                rgb = filter_pixel_nagao(u, v)
                self.set_pix(u, v, rgb)

        img_result = self.pixels_copy.reshape(self.height, self.width, 3)

        return img_result

    def filter_malik(self, iterations_param=2, alpha_param=0.2, kappa=25):
        """Perona Malik image filter

        Parameters
        ----------
        iterations_param : int
            count of iterations

        alpha_param : float
            Update rate (alpha)

        kappa : float
            Smoothness parameter (kappa)

        Returns
        ----------
        out : array with shape (height, width, 3)
            image data
        """

        m = self.width
        n = self.height

        t = iterations_param

        def eval_func(d):
            g_k = d / kappa

            return 1.0 / (1.0 + g_k*g_k)

        b = np.zeros((n, m))
        i_x = np.zeros((n, m, 3))
        i_y = np.zeros((n, m, 3))
        b_x = np.zeros((n, m))
        b_y = np.zeros((n, m))
        img_data = self.img_copy

        def copy_result_to_image(img_data):
            """copy filter result to image

            Parameters
            ----------
            img_data : array with shape(height, width, 3)
                image data
            """

            w = self.width
            h = self.height

            c = [0, 0, 0]

            for v in range(0, m):
                for u in range(0, n):
                    self.get_pixel_rgb(u, v, c)

                    c[0] = self.clamp(int(np.round(img_data[u][v][0])))
                    c[1] = self.clamp(int(np.round(img_data[u][v][1])))
                    c[2] = self.clamp(int(np.round(img_data[u][v][2])))

                    self.set_pix(u, v, c)

        def get_brightness(r, g, b):
            """copy filter result to image

            Parameters
            ----------
            r : int
                pixel red chanel

            g : int
                pixel green chanel

            b : int
                pixel blue chanel

            Returns
            ----------
            out : float
                brightness of pixel

            """

            return 0.299 * r + 0.587 * g + 0.114 * b

        def iterate_once():
            """main iteration of algorithm"""

            # пересчитаем показатели яркости
            for v in range(0, m):
                for u in range(0, n):
                    b[u][v] = get_brightness(img_data[u][v][0], img_data[u][v][1], img_data[u][v][2])

            # пересчитаем локальные разности цвета и цветового градиента по X и Y
            for v in range(0, m):
                for u in range(0, n):
                    if u < n - 1:
                        if img_data[u][v][0] < img_data[u + 1][v][0]:
                            i_x[u][v][0] = self.clamp(img_data[u + 1][v][0] - img_data[u][v][0])
                        else:
                            i_x[u][v][0] = 0
                        if img_data[u][v][1] < img_data[u + 1][v][1]:
                            i_x[u][v][1] = self.clamp(img_data[u + 1][v][1] - img_data[u][v][1])
                        else:
                            i_x[u][v][1] = 0
                        if img_data[u][v][2] < img_data[u + 1][v][2]:
                            i_x[u][v][2] = self.clamp(img_data[u + 1][v][2] - img_data[u][v][2])
                        else:
                            i_x[u][v][2] = 0

                        b_x[u][v] = b[u + 1][v] - b[u][v]
                    else:
                        i_x[u][v][0], i_x[u][v][1], i_x[u][v][2], b_x[u][v] = 0, 0, 0, 0
                    if v < m - 1:
                        if img_data[u][v][0] < img_data[u][v + 1][0]:
                            i_y[u][v][0] = self.clamp(img_data[u][v + 1][0] - img_data[u][v][0])
                        else:
                            i_y[u][v][0] = 0
                        if img_data[u][v][1] < img_data[u][v + 1][1]:
                            i_y[u][v][1] = self.clamp(img_data[u][v + 1][1] - img_data[u][v][1])
                        else:
                            i_y[u][v][1] = 0
                        if img_data[u][v][2] < img_data[u][v + 1][2]:
                            i_y[u][v][2] = self.clamp(img_data[u][v + 1][2] - img_data[u][v][2])
                        else:
                            i_y[u][v][2] = 0
                        b_y[u][v] = b[u][v + 1] - b[u][v]
                    else:
                        i_y[u][v][0], i_y[u][v][1], i_y[u][v][2], b_y[u][v] = 0, 0, 0, 0

            # обновим данные
            for v in range(0, m):
                for u in range(0, n):
                    if u > 0:
                        dw = -b_x[u - 1][v]
                    else:
                        dw = 0

                    de = b_x[u][v]
                    if v > 0:
                        dn = -b_y[u][v - 1]
                    else:
                        dn = 0

                    ds = b_y[u][v]
                    for i in range(0, 3):
                        if u > 0:
                            d_w_rgb = -i_x[u - 1][v][i]
                        else:
                            d_w_rgb = 0

                        d_e_rgb = i_x[u][v][i]

                        if v > 0:
                            d_n_rgb = -i_y[u][v - 1][i]
                        else:
                            d_n_rgb = 0

                        d_s_rgb = i_y[u][v][i]

                        img_data[u][v][i] = img_data[u][v][i] + alpha_param * (
                                eval_func(dn) * d_n_rgb + eval_func(ds) * d_s_rgb + eval_func(de) * d_e_rgb +
                                eval_func(dw) * d_w_rgb)

        for i in range(1, t + 1):
            iterate_once()

        copy_result_to_image(img_data)

        img_copy_perona_malik = self.pixels_copy.reshape(self.height, self.width, 3)

        return img_copy_perona_malik