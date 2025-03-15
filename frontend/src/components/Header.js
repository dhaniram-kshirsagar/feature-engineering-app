import React from 'react';
import { AppBar, Toolbar, Typography, Box, Button } from '@mui/material';
import { Link } from 'react-router-dom';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Box display="flex" alignItems="center" sx={{ flexGrow: 1 }}>
          <AutoFixHighIcon sx={{ mr: 1 }} />
          <Typography variant="h6" component={Link} to="/" style={{ textDecoration: 'none', color: 'white' }}>
            GenAI Feature Engineering
          </Typography>
        </Box>
        <Button color="inherit" component="a" href="https://github.com/business-science/ai-data-science-team" target="_blank">
          GitHub
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
