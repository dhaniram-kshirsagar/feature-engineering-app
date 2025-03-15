import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
  return (
    <Box 
      component="footer" 
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: (theme) => theme.palette.grey[100],
        textAlign: 'center'
      }}
    >
      <Typography variant="body2" color="text.secondary">
        {'© '}
        {new Date().getFullYear()}
        {' GenAI Feature Engineering App | Built with '}
        <Link color="inherit" href="https://github.com/business-science/ai-data-science-team" target="_blank">
          AI Data Science Team
        </Link>
      </Typography>
    </Box>
  );
};

export default Footer;
