-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: May 29, 2024 at 02:10 AM
-- Server version: 8.2.0
-- PHP Version: 8.2.13

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `portfolio_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `about_section`
--

DROP TABLE IF EXISTS `about_section`;
CREATE TABLE IF NOT EXISTS `about_section` (
  `id` int NOT NULL AUTO_INCREMENT,
  `profile_picture2` varchar(255) DEFAULT NULL,
  `biography` text,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `about_section`
--

INSERT INTO `about_section` (`id`, `profile_picture2`, `biography`) VALUES
(1, 'about-pic1.png', 'I am Gianne Bacay, AI Researcher, currently studying as a BSIT 2nd Year undergrad at the College of Information and Computing at the University of Southeastern Philippines, dedicated to pushing the boundaries of technology where I aim to achieve Artificial General Intelligence (AGI) and redefine human-computer interactions using cutting-edge AI advancements.');

-- --------------------------------------------------------

--
-- Table structure for table `contact_section`
--

DROP TABLE IF EXISTS `contact_section`;
CREATE TABLE IF NOT EXISTS `contact_section` (
  `id` int NOT NULL AUTO_INCREMENT,
  `gmail_link` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `contact_section`
--

INSERT INTO `contact_section` (`id`, `gmail_link`) VALUES
(1, 'giannebacay2004@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `credentials`
--

DROP TABLE IF EXISTS `credentials`;
CREATE TABLE IF NOT EXISTS `credentials` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `credentials`
--

INSERT INTO `credentials` (`id`, `username`, `password`) VALUES
(1, 'bacay', '$2a$10$CvVtVWmJfC/IP/XuB32UuOU7boXvdJiHw7NTnbjoqkMSo0G2F2i7m');

-- --------------------------------------------------------

--
-- Table structure for table `experience_section`
--

DROP TABLE IF EXISTS `experience_section`;
CREATE TABLE IF NOT EXISTS `experience_section` (
  `id` int NOT NULL AUTO_INCREMENT,
  `icon_path` varchar(255) NOT NULL,
  `technology_name` varchar(100) NOT NULL,
  `proficiency_level` varchar(50) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=105 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `experience_section`
--

INSERT INTO `experience_section` (`id`, `icon_path`, `technology_name`, `proficiency_level`) VALUES
(1, 'html.png', 'HTML', 'Basic'),
(2, 'css.png', 'CSS', 'Basic'),
(3, 'javascript.png', 'JavaScript', 'Basic'),
(4, 'php.png', 'PHP', 'Basic'),
(5, 'laravel.png', 'Laravel', 'Basic'),
(6, 'python.png', 'Python', 'Intermediate'),
(7, 'java.png', 'Java', 'Intermediate'),
(8, 'palm2.png', 'PaLM 2', 'Intermediate'),
(9, 'opencv.png', 'OpenCV', 'Basic'),
(10, 'mediapipe.png', 'MediaPipe', 'Basic'),
(11, 'vertexai.png', 'Vertex AI', 'Basic'),
(12, 'mysql.png', 'MySQL', 'Intermediate'),
(13, 'wampserver.png', 'Wamp server', 'Basic'),
(14, 'sqlyog.png', 'SQLyog', 'Basic'),
(15, 'phpmyadmin.png', 'phpMyAdmin', 'Intermediate'),
(16, 'googlecloud.png', 'Google Cloud', 'Basic'),
(17, 'githubdesktop.png', 'Github', 'Intermediate'),
(18, 'jira.png', 'Jira', 'Basic'),
(19, 'adobexd.png', 'Adobe Xd', 'Basic'),
(20, 'netbeans.png', 'Net Beans', 'Intermediate'),
(21, 'vsc.png', 'Visual Studio Code', 'Intermediate');

-- --------------------------------------------------------

--
-- Table structure for table `profile_section`
--

DROP TABLE IF EXISTS `profile_section`;
CREATE TABLE IF NOT EXISTS `profile_section` (
  `id` int NOT NULL AUTO_INCREMENT,
  `profile_picture1` varchar(255) DEFAULT NULL,
  `name` varchar(100) NOT NULL,
  `title` varchar(100) NOT NULL,
  `cv_link` varchar(255) DEFAULT NULL,
  `fb_link` varchar(255) DEFAULT NULL,
  `tt_link` varchar(255) DEFAULT NULL,
  `li_link` varchar(255) DEFAULT NULL,
  `gh_link` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `profile_section`
--

INSERT INTO `profile_section` (`id`, `profile_picture1`, `name`, `title`, `cv_link`, `fb_link`, `tt_link`, `li_link`, `gh_link`) VALUES
(1, 'profile-pic.png', 'Gianne Bacay', 'AI Researcher', 'resume.pdf', 'https://www.facebook.com/GianneBacay', 'https://www.tiktok.com/@gpbacay', 'https://www.linkedin.com/in/gianne-bacay-195527290/', 'https://github.com/gpbacay');

-- --------------------------------------------------------

--
-- Table structure for table `projects_section`
--

DROP TABLE IF EXISTS `projects_section`;
CREATE TABLE IF NOT EXISTS `projects_section` (
  `id` int NOT NULL AUTO_INCREMENT,
  `thumbnail` varchar(255) DEFAULT NULL,
  `projTitle` varchar(100) NOT NULL,
  `projDesc` text,
  `projGh` varchar(255) DEFAULT NULL,
  `projGdrive` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `projects_section`
--

INSERT INTO `projects_section` (`id`, `thumbnail`, `projTitle`, `projDesc`, `projGh`, `projGdrive`) VALUES
(1, 'project-1.png', 'H.A.R.A.Y.A', 'A personal AI assistant that integrates cutting-edge technologies such as computer vision, web data scraping, computer automation, natural language processing (NLP), and more.', 'https://github.com/gpbacay/PROJECT_H.A.R.A.Y.A.git', 'https://drive.google.com/drive/folders/1Ie63RLSyFu2rba0cxWoZIdQeHdSJqsPb?usp=sharing'),
(2, 'project-2.png', 'G.O.D.S.E.Y.E.S', 'An intelligent system designed to use advanced computer vision algorithms for face and pose detection.', 'https://github.com/gpbacay/PROJECT-G.O.D.S.E.Y.E.S.git', 'https://drive.google.com/drive/folders/13nPVQohEFfmCp4Mq65eRB3_zngs4sU3e?usp=sharing'),
(3, 'project-3.png', 'ADF I.B.M.S', 'A barbershop booking management system that simplifies appointments and enhances the customer experience.', 'https://github.com/gpbacay/Data_Structures/tree/3e4d95ca9316038df88fa1af2fecc2530d90923f/LE1/ADF_IBMS', 'https://drive.google.com/drive/folders/1jZvInutkbHSL0apMPt2M8YTLmWdrDDaS?usp=sharing'),
(4, 'project-4.png', 'CLINITRACK B.C.I.M.S', 'A platform for barangay clinics to track children\'s immunization, ensuring timely vaccinations.', 'https://github.com/gpbacay/Pogramming-Paradigm-2/tree/138eed8ef272475a55a917c403b74cb6a0c3c127/LE1', 'https://drive.google.com/drive/folders/1bLXSfHsxf1NPJoynW7Y9Wqtr0v18UT2l?usp=sharing');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
